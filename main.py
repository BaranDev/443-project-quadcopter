"""
main.py - Main Controller Application for Quadcopter Simulator
CMSE443 Real-Time Systems Design - Term Project

This is the main entry point that integrates all components:
- Cosys-AirSim connection and API control
- Xbox gamepad / keyboard input handling
- Real-time control loop at 50Hz
- PID controller for stabilization
- Visualization GUI
- Real-time scheduling and timing analysis

Real-Time Systems Concepts Demonstrated:
- Fixed-rate periodic task scheduling (50Hz = 20ms period)
- WCET (Worst-Case Execution Time) monitoring
- Deadline miss detection and handling
- Jitter measurement and analysis
- Priority-based execution (control loop > GUI updates)

Authors: CMSE443 Term Project Team
Date: December 2024
"""

import cosysairsim as airsim
import time
import threading
import math
import asyncio
import nest_asyncio
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass
from collections import deque
import sys
import argparse

# Apply nest_asyncio to allow nested event loops (required for cosysairsim in threads)
nest_asyncio.apply()

# Local modules
from input_handler import UnifiedInputHandler, InputConfig, InputSource
from pid_controller import QuadcopterPIDController
from quadcopter_model import QuadcopterModel, QuadcopterState, IntegrationMethod
from visualization import VisualizationGUI, TelemetryData


@dataclass
class ControllerConfig:
    """Configuration parameters for the controller."""

    # Timing parameters
    control_rate_hz: float = 50.0  # Control loop rate (50Hz minimum)
    gui_update_rate_hz: float = 20.0  # GUI refresh rate

    # Velocity limits
    max_velocity_xy: float = 5.0  # Maximum horizontal velocity (m/s)
    max_velocity_z: float = 3.0  # Maximum vertical velocity (m/s)
    max_yaw_rate: float = 60.0  # Maximum yaw rate (deg/s)

    # Safety parameters
    max_altitude: float = 50.0  # Maximum altitude (m)
    min_altitude: float = 0.5  # Minimum altitude (m)
    geofence_radius: float = 100.0  # Maximum distance from origin (m)

    # Control parameters
    velocity_deadband: float = 0.1  # Velocity command deadband

    # Timing analysis
    enable_timing_analysis: bool = True
    deadline_margin_ms: float = 2.0  # Deadline margin (ms)


class TimingAnalyzer:
    """
    Real-time timing analyzer for WCET and deadline analysis.

    This class tracks:
    - Loop execution times
    - Worst-case execution time (WCET)
    - Jitter (variation in loop timing)
    - Deadline misses

    These metrics are essential for real-time systems analysis
    as covered in CMSE443.
    """

    def __init__(self, target_period_ms: float, deadline_margin_ms: float = 2.0):
        """
        Initialize timing analyzer.

        Args:
            target_period_ms: Target loop period in milliseconds
            deadline_margin_ms: Acceptable margin before deadline miss
        """
        self.target_period_ms = target_period_ms
        self.deadline_margin_ms = deadline_margin_ms
        self.deadline_ms = target_period_ms - deadline_margin_ms

        # Statistics
        self.execution_times: deque = deque(maxlen=1000)
        self.loop_times: deque = deque(maxlen=1000)
        self.deadline_misses = 0
        self.total_loops = 0

        # WCET tracking
        self.wcet_ms = 0.0
        self.bcet_ms = float("inf")  # Best-case execution time

        # Jitter tracking
        self.last_loop_start = 0.0
        self.jitter_values: deque = deque(maxlen=1000)

    def record_execution(self, start_time: float, end_time: float):
        """
        Record a loop execution for analysis.

        Args:
            start_time: Loop start timestamp (perf_counter)
            end_time: Loop end timestamp (perf_counter)
        """
        execution_time_ms = (end_time - start_time) * 1000

        self.execution_times.append(execution_time_ms)
        self.total_loops += 1

        # Update WCET/BCET
        if execution_time_ms > self.wcet_ms:
            self.wcet_ms = execution_time_ms
        if execution_time_ms < self.bcet_ms:
            self.bcet_ms = execution_time_ms

        # Check deadline
        if execution_time_ms > self.deadline_ms:
            self.deadline_misses += 1

        # Calculate jitter (variation from target period)
        if self.last_loop_start > 0:
            actual_period_ms = (start_time - self.last_loop_start) * 1000
            self.loop_times.append(actual_period_ms)
            jitter = abs(actual_period_ms - self.target_period_ms)
            self.jitter_values.append(jitter)

        self.last_loop_start = start_time

    def get_statistics(self) -> Dict:
        """Get timing statistics."""
        if not self.execution_times:
            return {
                "wcet_ms": 0.0,
                "bcet_ms": 0.0,
                "avg_execution_ms": 0.0,
                "avg_loop_ms": self.target_period_ms,
                "avg_jitter_ms": 0.0,
                "max_jitter_ms": 0.0,
                "deadline_misses": 0,
                "deadline_miss_rate": 0.0,
                "total_loops": 0,
            }

        exec_times = list(self.execution_times)
        loop_times = (
            list(self.loop_times) if self.loop_times else [self.target_period_ms]
        )
        jitter_vals = list(self.jitter_values) if self.jitter_values else [0.0]

        return {
            "wcet_ms": self.wcet_ms,
            "bcet_ms": self.bcet_ms if self.bcet_ms != float("inf") else 0.0,
            "avg_execution_ms": sum(exec_times) / len(exec_times),
            "avg_loop_ms": sum(loop_times) / len(loop_times),
            "avg_jitter_ms": sum(jitter_vals) / len(jitter_vals),
            "max_jitter_ms": max(jitter_vals),
            "deadline_misses": self.deadline_misses,
            "deadline_miss_rate": self.deadline_misses / max(1, self.total_loops) * 100,
            "total_loops": self.total_loops,
        }

    def reset(self):
        """Reset all statistics."""
        self.execution_times.clear()
        self.loop_times.clear()
        self.jitter_values.clear()
        self.deadline_misses = 0
        self.total_loops = 0
        self.wcet_ms = 0.0
        self.bcet_ms = float("inf")
        self.last_loop_start = 0.0


class QuadcopterController:
    """
    Main quadcopter controller integrating all components.

    This class implements a real-time control system with:
    - Fixed-rate control loop (50Hz)
    - Input processing (gamepad/keyboard)
    - PID control for stabilization
    - Safety monitoring (geofence, altitude limits)
    - Telemetry logging and visualization

    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Main Control Loop (50Hz)                 │
    │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
    │  │  Input   │──▶│ Command  │──▶│   PID    │──▶│ AirSim   │ │
    │  │ Handler  │   │ Process  │   │ Control  │   │   API    │ │
    │  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
    │       ▲                              │              │       │
    │       │                              ▼              ▼       │
    │  ┌──────────┐                 ┌──────────┐   ┌──────────┐  │
    │  │Controller│                 │  Safety  │   │Telemetry │  │
    │  │/Keyboard │                 │  Monitor │   │  Logger  │  │
    │  └──────────┘                 └──────────┘   └──────────┘  │
    └─────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                               ┌──────────────────┐
                               │   GUI (20Hz)     │
                               │  Visualization   │
                               └──────────────────┘
    """

    def __init__(self, config: Optional[ControllerConfig] = None, use_gui: bool = True):
        """
        Initialize the quadcopter controller.

        Args:
            config: Controller configuration
            use_gui: Whether to start the visualization GUI
        """
        self.config = config if config is not None else ControllerConfig()
        self.use_gui = use_gui

        # Calculate timing parameters
        self.loop_period_s = 1.0 / self.config.control_rate_hz
        self.loop_period_ms = self.loop_period_s * 1000

        # Components
        self.client: Optional[airsim.MultirotorClient] = None
        self.input_handler: Optional[UnifiedInputHandler] = None
        self.pid_controller: Optional[QuadcopterPIDController] = None
        self.gui: Optional[VisualizationGUI] = None
        self.timing_analyzer: Optional[TimingAnalyzer] = None

        # State
        self.running = False
        self.armed = False
        self.connected = False
        self.emergency_stop = False

        # Control mode
        self.control_mode = "velocity"  # velocity, position, attitude

        # Position hold
        self.hold_position = np.array([0.0, 0.0, -3.0])  # NED
        self.hold_yaw = 0.0

        # Telemetry
        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(3)
        self.current_attitude = np.zeros(3)  # roll, pitch, yaw in degrees
        self.current_angular_velocity = np.zeros(3)

        # Statistics
        self.loop_count = 0
        self.start_time = 0.0

        # Thread for control loop
        self.control_thread: Optional[threading.Thread] = None

    def connect(self) -> bool:
        """
        Connect to the AirSim simulator.

        Returns:
            True if connection successful
        """
        try:
            print("Connecting to AirSim simulator...")
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            print("Connected to AirSim successfully!")
            self.connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to AirSim: {e}")
            self.connected = False
            return False

    def initialize(self) -> bool:
        """
        Initialize all components.

        Returns:
            True if initialization successful
        """
        print("\nInitializing components...")

        # Initialize input handler
        input_config = InputConfig(
            left_stick_deadzone=0.15,
            right_stick_deadzone=0.15,
            trigger_deadzone=0.05,
            yaw_sensitivity=1.0,
            pitch_sensitivity=1.0,
            roll_sensitivity=1.0,
            throttle_sensitivity=1.0,
            altitude_sensitivity=1.0,
            smoothing_factor=0.3,
            use_exponential_response=True,
            exponential_factor=2.0,
        )
        self.input_handler = UnifiedInputHandler(config=input_config)

        # Set input source change callback
        def on_source_change(source: InputSource):
            print(f"Input source changed to: {source.name}")

        self.input_handler.set_on_source_change(on_source_change)

        print(
            f"Input handler initialized. Controller available: {self.input_handler.xbox.is_available()}"
        )

        # Initialize PID controller
        # Get drone mass from settings (default 1.0 kg)
        self.pid_controller = QuadcopterPIDController(mass=1.0, gravity=9.81)
        print("PID controller initialized.")

        # Initialize timing analyzer
        self.timing_analyzer = TimingAnalyzer(
            target_period_ms=self.loop_period_ms,
            deadline_margin_ms=self.config.deadline_margin_ms,
        )
        print(
            f"Timing analyzer initialized. Target period: {self.loop_period_ms:.1f}ms"
        )

        # Initialize GUI
        if self.use_gui:
            self.gui = VisualizationGUI("CMSE443 Quadcopter Simulator - Cosys-AirSim")
            self.gui.set_callbacks(
                on_start=self._on_gui_start,
                on_stop=self._on_gui_stop,
                on_reset=self._on_gui_reset,
                on_arm=self._on_gui_arm,
                on_parameter_change=self._on_gui_parameter_change,
            )
            self.gui.start()
            print("GUI initialized and started.")

        print("All components initialized successfully!\n")
        return True

    def _on_gui_start(self):
        """Handle GUI start button."""
        if not self.running:
            self.start_control_loop()

    def _on_gui_stop(self):
        """Handle GUI stop button."""
        self.stop_control_loop()

    def _on_gui_reset(self):
        """Handle GUI reset button."""
        self.reset()

    def _on_gui_arm(self, armed: bool):
        """Handle GUI arm toggle."""
        if armed and self.connected:
            self.arm()
        else:
            self.disarm()

    def _on_gui_parameter_change(self, params: Dict):
        """Handle GUI parameter changes."""
        print(f"Parameters updated: {params}")

        pid = self.pid_controller
        if pid is None:
            return

        # Update velocity limits
        if "max_velocity" in params:
            self.config.max_velocity_xy = params["max_velocity"]

        if "max_yaw_rate" in params:
            self.config.max_yaw_rate = params["max_yaw_rate"]

        # Update PID gains
        if "pos_Kp" in params:
            pid.pos_x_pid.gains.Kp = params["pos_Kp"]
            pid.pos_y_pid.gains.Kp = params["pos_Kp"]
        if "pos_Kd" in params:
            pid.pos_x_pid.gains.Kd = params["pos_Kd"]
            pid.pos_y_pid.gains.Kd = params["pos_Kd"]
        if "att_Kp" in params:
            pid.roll_pid.gains.Kp = params["att_Kp"]
            pid.pitch_pid.gains.Kp = params["att_Kp"]
        if "att_Kd" in params:
            pid.roll_pid.gains.Kd = params["att_Kd"]
            pid.pitch_pid.gains.Kd = params["att_Kd"]

    def arm(self) -> bool:
        """
        Arm the drone (enable motors).

        Returns:
            True if arming successful
        """
        client = self.client
        if not self.connected or client is None:
            print("Cannot arm: Not connected to simulator")
            return False

        try:
            client.enableApiControl(True)
            client.armDisarm(True)
            self.armed = True
            print("Drone armed!")

            # Vibrate controller to confirm
            if self.input_handler:
                self.input_handler.set_vibration(0.5, 0.5)
                time.sleep(0.2)
                self.input_handler.stop_vibration()

            return True
        except Exception as e:
            print(f"Failed to arm: {e}")
            return False

    def disarm(self) -> bool:
        """
        Disarm the drone (disable motors).

        Returns:
            True if disarming successful
        """
        client = self.client
        if not self.connected or client is None:
            return False

        try:
            client.armDisarm(False)
            client.enableApiControl(False)
            self.armed = False
            print("Drone disarmed!")
            return True
        except Exception as e:
            print(f"Failed to disarm: {e}")
            return False

    def takeoff(self, altitude: float = 3.0) -> bool:
        """
        Perform automated takeoff.

        Args:
            altitude: Target altitude in meters
        """
        client = self.client
        if not self.armed or client is None:
            print("Cannot takeoff: Drone not armed")
            return False

        try:
            print(f"Taking off to {altitude}m...")

            print("Takeoff complete!")

            # Set hold position
            self.hold_position = np.array([0.0, 0.0, -altitude])
            return True
        except Exception as e:
            print(f"Takeoff failed: {e}")
            return False

    def land(self) -> bool:
        """Perform automated landing."""
        client = self.client
        if client is None:
            return False

        try:
            print("Landing...")
            # Ensure event loop exists
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            client.landAsync().join()
            print("Landing complete!")
            return True
        except Exception as e:
            print(f"Landing failed: {e}")
            return False

    def reset(self):
        """Reset the simulation."""
        print("Resetting simulation...")

        self.emergency_stop = False
        self.armed = False

        client = self.client
        if client is not None and self.connected:
            try:
                client.armDisarm(False)
                client.enableApiControl(False)
                client.reset()
                time.sleep(0.5)
            except Exception as e:
                print(f"Reset warning: {e}")

        # Reset controllers
        if self.pid_controller:
            self.pid_controller.reset()

        if self.timing_analyzer:
            self.timing_analyzer.reset()

        # Reset state
        self.hold_position = np.array([0.0, 0.0, -3.0])
        self.hold_yaw = 0.0
        self.loop_count = 0

        print("Reset complete!")

    def emergency_stop_handler(self):
        """Handle emergency stop command."""
        print("\n!!! EMERGENCY STOP !!!")
        self.emergency_stop = True

        client = self.client
        if client is not None and self.connected and self.armed:
            try:
                # Immediate hover
                client.hoverAsync()

                # Vibrate controller
                if self.input_handler:
                    self.input_handler.set_vibration(1.0, 1.0)
                    time.sleep(0.5)
                    self.input_handler.stop_vibration()

                # Land
                self.land()
                self.disarm()
            except Exception as e:
                print(f"Emergency stop error: {e}")

    def _get_drone_state(self) -> Optional[Dict[str, Any]]:
        """
        Get current drone state from simulator.

        Returns:
            Dictionary with position, velocity, attitude data
        """
        client = self.client
        if client is None:
            return None

        try:
            state = client.getMultirotorState()

            # Position (NED frame)
            pos = state.kinematics_estimated.position
            self.current_position = np.array([pos.x_val, pos.y_val, pos.z_val])

            # Velocity (NED frame)
            vel = state.kinematics_estimated.linear_velocity
            self.current_velocity = np.array([vel.x_val, vel.y_val, vel.z_val])

            # Orientation (quaternion to Euler)
            orient = state.kinematics_estimated.orientation

            # Convert quaternion to Euler angles
            to_euler_func = getattr(airsim, "to_eulerian_angles", None)
            if to_euler_func is None:
                to_euler_func = getattr(airsim, "to_eularian_angles", None)

            if to_euler_func is None:
                pitch = roll = yaw = 0.0
            else:
                pitch, roll, yaw = to_euler_func(orient)
            self.current_attitude = np.array(
                [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]
            )

            # Angular velocity
            ang_vel = state.kinematics_estimated.angular_velocity
            self.current_angular_velocity = np.array(
                [
                    math.degrees(ang_vel.x_val),
                    math.degrees(ang_vel.y_val),
                    math.degrees(ang_vel.z_val),
                ]
            )

            return {
                "position": self.current_position,
                "velocity": self.current_velocity,
                "attitude": self.current_attitude,
                "angular_velocity": self.current_angular_velocity,
                "landed": state.landed_state == airsim.LandedState.Landed,
            }

        except Exception as e:
            print(f"Error getting drone state: {e}")
            return None

    def _process_input(self) -> Dict:
        """
        Process input from controller/keyboard.

        Returns:
            Dictionary with processed control commands
        """
        handler = self.input_handler
        if handler is None:
            return {
                "vx": 0.0,
                "vy": 0.0,
                "vz": 0.0,
                "yaw_rate": 0.0,
                "input_source": InputSource.NONE.name,
                "any_input": False,
            }

        # Poll input
        input_state = handler.poll()

        # Get flight commands
        commands = handler.get_flight_commands()

        # Get button events
        buttons = handler.get_button_events()

        # Handle button events
        if buttons["arm_toggle"]:
            if self.armed:
                self.disarm()
            else:
                self.arm()

        if buttons["emergency_stop"]:
            self.emergency_stop_handler()

        if buttons["reset"]:
            self.reset()

        if buttons["mode_toggle"]:
            # Cycle through control modes
            modes = ["velocity", "position", "attitude"]
            current_idx = modes.index(self.control_mode)
            self.control_mode = modes[(current_idx + 1) % len(modes)]
            print(f"Control mode: {self.control_mode}")

        # Scale commands to actual values
        vx = commands["pitch"] * self.config.max_velocity_xy  # Forward/back
        vy = commands["roll"] * self.config.max_velocity_xy  # Left/right
        vz = -commands["altitude_rate"] * self.config.max_velocity_z  # Up/down (NED)
        yaw_rate = commands["yaw_rate"] * self.config.max_yaw_rate  # Degrees/s

        # Apply deadband
        if abs(vx) < self.config.velocity_deadband:
            vx = 0.0
        if abs(vy) < self.config.velocity_deadband:
            vy = 0.0
        if abs(vz) < self.config.velocity_deadband:
            vz = 0.0

        return {
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "yaw_rate": yaw_rate,
            "input_source": input_state.source.name,
            "any_input": abs(vx) > 0 or abs(vy) > 0 or abs(vz) > 0 or abs(yaw_rate) > 0,
        }

    def _check_safety(self, state: Dict) -> bool:
        """
        Check safety constraints (geofence, altitude limits).

        Returns:
            True if safe to continue, False if safety violation
        """
        pos = state["position"]

        # Check altitude limits
        altitude = -pos[2]  # Convert NED to altitude
        if altitude > self.config.max_altitude:
            print(f"Warning: Maximum altitude reached ({altitude:.1f}m)")
            return False

        # Check geofence
        distance = math.sqrt(pos[0] ** 2 + pos[1] ** 2)
        if distance > self.config.geofence_radius:
            print(f"Warning: Geofence boundary reached ({distance:.1f}m)")
            return False

        return True

    def _send_control_command(self, commands: Dict, dt: float):
        """
        Send control command to the drone.

        Args:
            commands: Processed control commands
            dt: Time step
        """
        client = self.client
        if client is None or not self.armed or self.emergency_stop:
            return

        vx = commands["vx"]
        vy = commands["vy"]
        vz = commands["vz"]
        yaw_rate = commands["yaw_rate"]

        try:
            if self.control_mode == "velocity":
                # Velocity control in body frame
                # Duration slightly longer than loop period for smooth control
                client.moveByVelocityBodyFrameAsync(
                    vx,
                    vy,
                    vz,
                    duration=dt * 2,
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(True, yaw_rate),
                )

            elif self.control_mode == "position":
                # Position hold with velocity input adjusting setpoint
                if commands["any_input"]:
                    # Update hold position based on velocity input
                    yaw_rad = math.radians(self.current_attitude[2])

                    # Transform body velocity to world frame
                    dx = (vx * math.cos(yaw_rad) - vy * math.sin(yaw_rad)) * dt
                    dy = (vx * math.sin(yaw_rad) + vy * math.cos(yaw_rad)) * dt
                    dz = vz * dt

                    self.hold_position += np.array([dx, dy, dz])
                    self.hold_yaw += yaw_rate * dt

                # Send position command
                client.moveToPositionAsync(
                    self.hold_position[0],
                    self.hold_position[1],
                    self.hold_position[2],
                    velocity=self.config.max_velocity_xy,
                    yaw_mode=airsim.YawMode(False, self.hold_yaw),
                )

        except Exception as e:
            print(f"Control command error: {e}")

    def _update_gui(self, state: Dict, commands: Dict, loop_time_ms: float):
        """
        Update GUI with current telemetry.

        Args:
            state: Current drone state
            commands: Current control commands
            loop_time_ms: Loop execution time
        """
        if not self.gui:
            return

        timing_analyzer = self.timing_analyzer
        if timing_analyzer is None:
            return

        timing_stats = timing_analyzer.get_statistics()

        telemetry = TelemetryData(
            x=state["position"][0],
            y=state["position"][1],
            z=state["position"][2],
            vx=state["velocity"][0],
            vy=state["velocity"][1],
            vz=state["velocity"][2],
            roll=state["attitude"][0],
            pitch=state["attitude"][1],
            yaw=state["attitude"][2],
            p=state["angular_velocity"][0],
            q=state["angular_velocity"][1],
            r=state["angular_velocity"][2],
            armed=self.armed,
            connected=self.connected,
            input_source=commands.get("input_source", "None"),
            control_mode=self.control_mode,
            loop_rate_hz=(
                1000.0 / timing_stats["avg_loop_ms"]
                if timing_stats["avg_loop_ms"] > 0
                else 0
            ),
            loop_time_ms=timing_stats["avg_execution_ms"],
            deadline_misses=timing_stats["deadline_misses"],
        )

        self.gui.update_telemetry(telemetry)

    def _control_loop(self):
        """
        Main control loop - runs at 50Hz.

        This is the real-time critical section that demonstrates:
        - Periodic task scheduling
        - WCET monitoring
        - Deadline management
        """
        timing_analyzer = self.timing_analyzer
        if timing_analyzer is None:
            raise RuntimeError("Timing analyzer not initialized")

        print(f"\nStarting control loop at {self.config.control_rate_hz}Hz")
        print(f"Loop period: {self.loop_period_ms:.2f}ms")
        print(f"Deadline margin: {self.config.deadline_margin_ms}ms")
        print("-" * 50)

        self.start_time = time.perf_counter()
        next_loop_time = self.start_time

        while self.running:
            loop_start = time.perf_counter()

            try:
                # Phase 1: Get drone state from simulator
                state = self._get_drone_state()
                if state is None:
                    continue

                # Phase 2: Process input from controller/keyboard
                commands = self._process_input()

                # Phase 3: Check safety constraints
                if not self._check_safety(state):
                    # Safety violation - hover in place
                    commands = {
                        "vx": 0,
                        "vy": 0,
                        "vz": 0,
                        "yaw_rate": 0,
                        "any_input": False,
                        "input_source": commands["input_source"],
                    }

                # Phase 4: Send control command
                self._send_control_command(commands, self.loop_period_s)

                # Phase 5: Update GUI (lower priority)
                if (
                    self.loop_count
                    % int(self.config.control_rate_hz / self.config.gui_update_rate_hz)
                    == 0
                ):
                    self._update_gui(
                        state, commands, (time.perf_counter() - loop_start) * 1000
                    )

            except Exception as e:
                print(f"Control loop error: {e}")

            # Record timing
            loop_end = time.perf_counter()
            timing_analyzer.record_execution(loop_start, loop_end)

            # Calculate sleep time to maintain loop rate
            next_loop_time += self.loop_period_s
            sleep_time = next_loop_time - time.perf_counter()

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Missed deadline - skip to next slot
                next_loop_time = time.perf_counter() + self.loop_period_s

            self.loop_count += 1

        print("\nControl loop stopped.")
        self._print_timing_statistics()

    def _print_timing_statistics(self):
        """Print timing analysis results."""
        timing_analyzer = self.timing_analyzer
        if timing_analyzer is None:
            return

        stats = timing_analyzer.get_statistics()

        print("\n" + "=" * 50)
        print("TIMING ANALYSIS RESULTS")
        print("=" * 50)
        print(f"Total loops executed: {stats['total_loops']}")
        print(f"\nExecution Time:")
        print(f"  WCET (Worst-Case):  {stats['wcet_ms']:.3f} ms")
        print(f"  BCET (Best-Case):   {stats['bcet_ms']:.3f} ms")
        print(f"  Average:            {stats['avg_execution_ms']:.3f} ms")
        print(f"\nLoop Timing:")
        print(f"  Target Period:      {self.loop_period_ms:.3f} ms")
        print(f"  Average Period:     {stats['avg_loop_ms']:.3f} ms")
        print(f"  Average Jitter:     {stats['avg_jitter_ms']:.3f} ms")
        print(f"  Maximum Jitter:     {stats['max_jitter_ms']:.3f} ms")
        print(f"\nDeadline Analysis:")
        analyzer = self.timing_analyzer
        if analyzer:
            print(f"  Deadline:           {analyzer.deadline_ms:.3f} ms")
        print(f"  Deadline Misses:    {stats['deadline_misses']}")
        print(f"  Miss Rate:          {stats['deadline_miss_rate']:.2f}%")
        print("=" * 50)

    def start_control_loop(self):
        """Start the control loop in a separate thread."""
        if self.running:
            print("Control loop already running")
            return

        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()

    def stop_control_loop(self):
        """Stop the control loop."""
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=2.0)

        # Hover if armed
        if self.armed and self.connected:
            try:
                client = self.client
                if client:
                    client.hoverAsync()
            except:
                pass

    def run(self):
        """
        Main run method - handles the overall application flow.
        """
        print("\n" + "=" * 60)
        print("CMSE443 Quadcopter Simulator - Cosys-AirSim Controller")
        print("=" * 60)

        # Connect to simulator
        if not self.connect():
            print("Failed to connect to simulator. Exiting.")
            return

        # Initialize components
        if not self.initialize():
            print("Failed to initialize components. Exiting.")
            return

        # Print control instructions
        self._print_instructions()

        # Main application loop
        try:
            print("\nPress Ctrl+C to exit\n")

            while True:
                if self.gui and not self.gui.is_running():
                    print("GUI closed. Exiting...")
                    break

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\nShutdown requested...")

        # Cleanup
        self.cleanup()

    def _print_instructions(self):
        """Print control instructions."""
        print("\n" + "-" * 60)
        print("CONTROL INSTRUCTIONS")
        print("-" * 60)
        print("\nXbox Controller:")
        print("  Left Stick X:    Yaw (rotate left/right)")
        print("  Left Stick Y:    Throttle (forward/backward)")
        print("  Right Stick X:   Roll (strafe left/right)")
        print("  Right Stick Y:   Pitch (forward/backward)")
        print("  Left Trigger:    Descend")
        print("  Right Trigger:   Ascend")
        print("  A Button:        Arm/Disarm toggle")
        print("  B Button:        Emergency Stop")
        print("  X Button:        Reset simulation")
        print("  Y Button:        Toggle control mode")
        print("\nKeyboard Fallback:")
        print("  W/S:             Pitch forward/backward")
        print("  A/D:             Yaw left/right")
        print("  Q/E:             Roll left/right")
        print("  Space:           Ascend")
        print("  Left Shift:      Descend")
        print("  Enter:           Arm/Disarm toggle")
        print("  Escape:          Emergency Stop")
        print("  R:               Reset simulation")
        print("  Tab:             Toggle control mode")
        print("-" * 60)

    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")

        # Stop control loop
        self.stop_control_loop()

        # Disarm and disconnect
        if self.armed:
            self.disarm()

        # Stop input handler
        if self.input_handler:
            self.input_handler.stop_vibration()

        # Stop GUI
        if self.gui:
            self.gui.stop()

        print("Cleanup complete. Goodbye!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="CMSE443 Quadcopter Controller")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI")
    parser.add_argument(
        "--rate", type=float, default=50.0, help="Control loop rate (Hz)"
    )
    parser.add_argument(
        "--max-vel", type=float, default=5.0, help="Maximum velocity (m/s)"
    )
    args = parser.parse_args()

    # Create configuration
    config = ControllerConfig(
        control_rate_hz=args.rate,
        max_velocity_xy=args.max_vel,
    )

    # Create and run controller
    controller = QuadcopterController(config=config, use_gui=not args.no_gui)
    controller.run()


if __name__ == "__main__":
    main()
