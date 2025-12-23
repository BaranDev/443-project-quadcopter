"""
pid_controller.py - PID Controller Implementation for Quadcopter
CMSE443 Real-Time Systems Design - Term Project

This module implements PID (Proportional-Integral-Derivative) controllers
for quadcopter stabilization and position control.

The controller hierarchy consists of:
1. Position Controller (outer loop) - commands desired attitude
2. Attitude Controller (inner loop) - commands motor thrust/moments
3. Rate Controller (innermost loop) - stabilizes angular rates

Reference:
- Åström, K. J., & Murray, R. M. (2010). Feedback systems: An introduction
  for scientists and engineers. Princeton University Press.
- Bouabdallah, S. (2007). Design and control of quadrotors with application
  to autonomous flying. EPFL PhD Thesis.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict
import time


@dataclass
class PIDGains:
    """
    PID controller gains.

    Standard PID control law:
    u(t) = Kp * e(t) + Ki * ∫e(τ)dτ + Kd * de(t)/dt

    Where:
    - Kp: Proportional gain (response to current error)
    - Ki: Integral gain (eliminates steady-state error)
    - Kd: Derivative gain (dampens oscillations)
    """

    Kp: float = 1.0  # Proportional gain
    Ki: float = 0.0  # Integral gain
    Kd: float = 0.0  # Derivative gain

    # Anti-windup limits for integral term
    integral_limit: float = 10.0

    # Output limits
    output_min: float = -float("inf")
    output_max: float = float("inf")

    # Derivative filter coefficient (0-1, higher = more filtering)
    derivative_filter: float = 0.1


@dataclass
class PIDState:
    """Internal state of a PID controller."""

    integral: float = 0.0  # Accumulated integral
    previous_error: float = 0.0  # Error from last step
    previous_derivative: float = 0.0  # Filtered derivative
    previous_time: float = 0.0  # Timestamp of last update
    initialized: bool = False


class PIDController:
    """
    Single-axis PID controller with anti-windup and derivative filtering.

    Features:
    - Anti-windup: Prevents integral term from growing unbounded
    - Derivative filtering: Low-pass filter to reduce noise sensitivity
    - Output limiting: Constrains output to specified range
    - Bumpless transfer: Smooth transitions when parameters change

    The derivative term uses the "derivative on measurement" technique
    to avoid spikes when the setpoint changes.
    """

    def __init__(self, gains: Optional[PIDGains] = None, name: str = "PID"):
        """
        Initialize PID controller.

        Args:
            gains: PID gains (uses defaults if None)
            name: Identifier for logging/debugging
        """
        self.gains = gains if gains is not None else PIDGains()
        self.name = name
        self.state = PIDState()

        # Performance metrics
        self.total_updates = 0
        self.total_error_squared = 0.0

    def reset(self):
        """Reset controller state."""
        self.state = PIDState()
        self.total_updates = 0
        self.total_error_squared = 0.0

    def set_gains(
        self,
        Kp: Optional[float] = None,
        Ki: Optional[float] = None,
        Kd: Optional[float] = None,
    ):
        """Update individual gains at runtime."""
        if Kp is not None:
            self.gains.Kp = Kp
        if Ki is not None:
            self.gains.Ki = Ki
        if Kd is not None:
            self.gains.Kd = Kd

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        """
        Compute PID control output.

        Args:
            setpoint: Desired value
            measurement: Current measured value
            dt: Time step since last update [s]

        Returns:
            Control output
        """
        # Calculate error
        error = setpoint - measurement

        # Initialize state on first call
        if not self.state.initialized:
            self.state.previous_error = error
            self.state.previous_time = time.perf_counter()
            self.state.initialized = True

        # Proportional term
        P = self.gains.Kp * error

        # Integral term with anti-windup
        self.state.integral += error * dt
        self.state.integral = np.clip(
            self.state.integral, -self.gains.integral_limit, self.gains.integral_limit
        )
        I = self.gains.Ki * self.state.integral

        # Derivative term with filtering (derivative on error)
        if dt > 0:
            derivative = (error - self.state.previous_error) / dt
            # Low-pass filter on derivative
            alpha = self.gains.derivative_filter
            filtered_derivative = (
                alpha * self.state.previous_derivative + (1 - alpha) * derivative
            )
            self.state.previous_derivative = filtered_derivative
        else:
            filtered_derivative = self.state.previous_derivative

        D = self.gains.Kd * filtered_derivative

        # Compute total output
        output = P + I + D

        # Apply output limits
        output = np.clip(output, self.gains.output_min, self.gains.output_max)

        # Update state
        self.state.previous_error = error

        # Update metrics
        self.total_updates += 1
        self.total_error_squared += error**2

        return output

    def get_rmse(self) -> float:
        """Get root mean squared error over all updates."""
        if self.total_updates == 0:
            return 0.0
        return np.sqrt(self.total_error_squared / self.total_updates)

    def get_components(
        self, setpoint: float, measurement: float, dt: float
    ) -> Tuple[float, float, float, float]:
        """
        Get individual P, I, D components for debugging/tuning.

        Returns:
            Tuple of (P, I, D, total) values
        """
        error = setpoint - measurement

        P = self.gains.Kp * error
        I = self.gains.Ki * self.state.integral

        if dt > 0:
            derivative = (error - self.state.previous_error) / dt
        else:
            derivative = 0
        D = self.gains.Kd * derivative

        total = P + I + D
        return P, I, D, total


class QuadcopterPIDController:
    """
    Complete PID control system for quadcopter stabilization.

    This implements a cascaded control architecture:

    ┌─────────────────────────────────────────────────────────────────┐
    │  Position Controller (Outer Loop)                               │
    │  ┌──────────┐   ┌──────────┐   ┌──────────┐                    │
    │  │ X-pos PID│   │ Y-pos PID│   │ Z-pos PID│                    │
    │  └────┬─────┘   └────┬─────┘   └────┬─────┘                    │
    │       │              │              │                          │
    │       ▼              ▼              ▼                          │
    │  pitch_cmd      roll_cmd       thrust_cmd                      │
    └─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  Attitude Controller (Inner Loop)                               │
    │  ┌──────────┐   ┌──────────┐   ┌──────────┐                    │
    │  │ Roll PID │   │Pitch PID │   │ Yaw PID  │                    │
    │  └────┬─────┘   └────┬─────┘   └────┬─────┘                    │
    │       │              │              │                          │
    │       ▼              ▼              ▼                          │
    │  tau_phi        tau_theta       tau_psi                        │
    └─────────────────────────────────────────────────────────────────┘

    The outer position loop runs at a slower rate (10-20 Hz) while
    the inner attitude loop runs at the full control rate (50-100 Hz).
    """

    # Default gains tuned for stable flight
    DEFAULT_POSITION_GAINS = {
        "x": PIDGains(Kp=2.0, Ki=0.1, Kd=1.5, output_min=-0.5, output_max=0.5),
        "y": PIDGains(Kp=2.0, Ki=0.1, Kd=1.5, output_min=-0.5, output_max=0.5),
        "z": PIDGains(Kp=4.0, Ki=0.5, Kd=2.0, output_min=-20.0, output_max=20.0),
    }

    DEFAULT_ATTITUDE_GAINS = {
        "roll": PIDGains(Kp=6.0, Ki=0.0, Kd=1.2, output_min=-5.0, output_max=5.0),
        "pitch": PIDGains(Kp=6.0, Ki=0.0, Kd=1.2, output_min=-5.0, output_max=5.0),
        "yaw": PIDGains(Kp=4.0, Ki=0.1, Kd=0.5, output_min=-2.0, output_max=2.0),
    }

    DEFAULT_RATE_GAINS = {
        "roll_rate": PIDGains(Kp=0.5, Ki=0.0, Kd=0.01),
        "pitch_rate": PIDGains(Kp=0.5, Ki=0.0, Kd=0.01),
        "yaw_rate": PIDGains(Kp=0.3, Ki=0.0, Kd=0.01),
    }

    def __init__(self, mass: float = 1.0, gravity: float = 9.81):
        """
        Initialize the quadcopter PID controller.

        Args:
            mass: Quadcopter mass in kg (for thrust feedforward)
            gravity: Gravitational acceleration in m/s²
        """
        self.mass = mass
        self.gravity = gravity
        self.hover_thrust = mass * gravity

        # Position controllers (outer loop)
        self.pos_x_pid = PIDController(self.DEFAULT_POSITION_GAINS["x"], "pos_x")
        self.pos_y_pid = PIDController(self.DEFAULT_POSITION_GAINS["y"], "pos_y")
        self.pos_z_pid = PIDController(self.DEFAULT_POSITION_GAINS["z"], "pos_z")

        # Attitude controllers (inner loop)
        self.roll_pid = PIDController(self.DEFAULT_ATTITUDE_GAINS["roll"], "roll")
        self.pitch_pid = PIDController(self.DEFAULT_ATTITUDE_GAINS["pitch"], "pitch")
        self.yaw_pid = PIDController(self.DEFAULT_ATTITUDE_GAINS["yaw"], "yaw")

        # Rate controllers (innermost loop)
        self.roll_rate_pid = PIDController(
            self.DEFAULT_RATE_GAINS["roll_rate"], "roll_rate"
        )
        self.pitch_rate_pid = PIDController(
            self.DEFAULT_RATE_GAINS["pitch_rate"], "pitch_rate"
        )
        self.yaw_rate_pid = PIDController(
            self.DEFAULT_RATE_GAINS["yaw_rate"], "yaw_rate"
        )

        # Control outputs
        self.thrust = self.hover_thrust
        self.tau_phi = 0.0  # Roll moment
        self.tau_theta = 0.0  # Pitch moment
        self.tau_psi = 0.0  # Yaw moment

        # Setpoints
        self.position_setpoint = np.array([0.0, 0.0, 0.0])  # [x, y, z] in NED
        self.yaw_setpoint = 0.0

        # Control modes
        self.position_control_enabled = True
        self.attitude_control_enabled = True
        self.rate_control_enabled = True

    def reset(self):
        """Reset all controllers."""
        self.pos_x_pid.reset()
        self.pos_y_pid.reset()
        self.pos_z_pid.reset()
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.yaw_pid.reset()
        self.roll_rate_pid.reset()
        self.pitch_rate_pid.reset()
        self.yaw_rate_pid.reset()

        self.thrust = self.hover_thrust
        self.tau_phi = 0.0
        self.tau_theta = 0.0
        self.tau_psi = 0.0

    def set_position_setpoint(self, x: float, y: float, z: float):
        """Set target position in NED frame."""
        self.position_setpoint = np.array([x, y, z])

    def set_yaw_setpoint(self, yaw: float):
        """Set target yaw angle in radians."""
        self.yaw_setpoint = yaw

    def update_position_control(
        self, position: np.ndarray, velocity: np.ndarray, dt: float
    ) -> Tuple[float, float, float]:
        """
        Position control loop (outer loop).

        Computes desired pitch and roll angles based on position error.
        Uses velocity feedforward for better tracking.

        Args:
            position: Current position [x, y, z] in NED frame
            velocity: Current velocity [vx, vy, vz]
            dt: Time step

        Returns:
            Tuple of (desired_pitch, desired_roll, thrust_adjustment)
        """
        if not self.position_control_enabled:
            return 0.0, 0.0, 0.0

        # Position errors
        error_x = self.position_setpoint[0] - position[0]
        error_y = self.position_setpoint[1] - position[1]
        error_z = self.position_setpoint[2] - position[2]

        # PID outputs (desired accelerations)
        accel_x_cmd = self.pos_x_pid.update(self.position_setpoint[0], position[0], dt)
        accel_y_cmd = self.pos_y_pid.update(self.position_setpoint[1], position[1], dt)
        # For Z in NED: negative z is UP, so we negate the output
        # When drone is below target (position_z > setpoint_z), we need more thrust
        thrust_adj = -self.pos_z_pid.update(self.position_setpoint[2], position[2], dt)

        # Convert desired accelerations to attitude commands
        # For small angles: pitch ≈ accel_x / g, roll ≈ -accel_y / g
        desired_pitch = np.clip(accel_x_cmd / self.gravity, -0.5, 0.5)  # rad
        desired_roll = np.clip(-accel_y_cmd / self.gravity, -0.5, 0.5)  # rad

        return desired_pitch, desired_roll, thrust_adj

    def update_attitude_control(
        self,
        attitude: np.ndarray,
        angular_velocity: np.ndarray,
        desired_roll: float,
        desired_pitch: float,
        dt: float,
    ) -> Tuple[float, float, float]:
        """
        Attitude control loop (inner loop).

        Computes desired angular rates based on attitude error.

        Args:
            attitude: Current Euler angles [phi, theta, psi] in radians
            angular_velocity: Current angular rates [p, q, r]
            desired_roll: Commanded roll angle
            desired_pitch: Commanded pitch angle
            dt: Time step

        Returns:
            Tuple of (tau_phi, tau_theta, tau_psi) moments
        """
        if not self.attitude_control_enabled:
            return 0.0, 0.0, 0.0

        phi, theta, psi = attitude
        p, q, r = angular_velocity

        # Attitude PID outputs (desired angular rates)
        roll_rate_cmd = self.roll_pid.update(desired_roll, phi, dt)
        pitch_rate_cmd = self.pitch_pid.update(desired_pitch, theta, dt)
        yaw_rate_cmd = self.yaw_pid.update(self.yaw_setpoint, psi, dt)

        if self.rate_control_enabled:
            # Rate control (innermost loop)
            tau_phi = self.roll_rate_pid.update(roll_rate_cmd, p, dt)
            tau_theta = self.pitch_rate_pid.update(pitch_rate_cmd, q, dt)
            tau_psi = self.yaw_rate_pid.update(yaw_rate_cmd, r, dt)
        else:
            # Direct attitude control
            tau_phi = roll_rate_cmd
            tau_theta = pitch_rate_cmd
            tau_psi = yaw_rate_cmd

        return tau_phi, tau_theta, tau_psi

    def compute_control(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        attitude: np.ndarray,
        angular_velocity: np.ndarray,
        dt: float,
    ) -> Dict[str, float]:
        """
        Full control computation (position + attitude + rate).

        This is the main control function called each control cycle.

        Args:
            position: Current position [x, y, z] in NED
            velocity: Current velocity [vx, vy, vz]
            attitude: Current Euler angles [phi, theta, psi]
            angular_velocity: Current angular rates [p, q, r]
            dt: Time step

        Returns:
            Dictionary with thrust, tau_phi, tau_theta, tau_psi
        """
        # Position control (outer loop)
        desired_pitch, desired_roll, thrust_adj = self.update_position_control(
            position, velocity, dt
        )

        # Attitude control (inner loop)
        tau_phi, tau_theta, tau_psi = self.update_attitude_control(
            attitude, angular_velocity, desired_roll, desired_pitch, dt
        )

        # Compute total thrust
        # Thrust = hover_thrust + altitude_adjustment
        # Adjusted for attitude: T_actual = T / cos(phi) / cos(theta)
        phi, theta = attitude[0], attitude[1]
        cos_correction = max(0.5, np.cos(phi) * np.cos(theta))
        self.thrust = (self.hover_thrust + thrust_adj) / cos_correction
        self.thrust = np.clip(self.thrust, 0, 2 * self.hover_thrust)

        self.tau_phi = tau_phi
        self.tau_theta = tau_theta
        self.tau_psi = tau_psi

        return {
            "thrust": self.thrust,
            "tau_phi": self.tau_phi,
            "tau_theta": self.tau_theta,
            "tau_psi": self.tau_psi,
        }

    def compute_velocity_control(
        self,
        velocity_cmd: np.ndarray,
        current_velocity: np.ndarray,
        attitude: np.ndarray,
        angular_velocity: np.ndarray,
        dt: float,
    ) -> Dict[str, float]:
        """
        Velocity control mode for manual flight.

        Instead of tracking a position setpoint, this tracks a velocity
        command, which is more natural for joystick control.

        Args:
            velocity_cmd: Commanded velocity [vx, vy, vz] in body frame
            current_velocity: Current velocity in world frame
            attitude: Current attitude [phi, theta, psi]
            angular_velocity: Current angular rates [p, q, r]
            dt: Time step

        Returns:
            Control outputs dictionary
        """
        phi, theta, psi = attitude

        # Transform commanded velocity from body to world frame
        cp, sp = np.cos(phi), np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        cy, sy = np.cos(psi), np.sin(psi)

        # Simplified rotation (assuming small roll/pitch)
        vx_world_cmd = velocity_cmd[0] * cy - velocity_cmd[1] * sy
        vy_world_cmd = velocity_cmd[0] * sy + velocity_cmd[1] * cy
        vz_world_cmd = velocity_cmd[2]

        # Velocity error
        vel_error_x = vx_world_cmd - current_velocity[0]
        vel_error_y = vy_world_cmd - current_velocity[1]
        vel_error_z = vz_world_cmd - current_velocity[2]

        # Simple P-control on velocity error to get attitude commands
        Kv = 0.3  # Velocity gain
        desired_pitch = np.clip(Kv * vel_error_x, -0.3, 0.3)
        desired_roll = np.clip(-Kv * vel_error_y, -0.3, 0.3)
        thrust_adj = -2.0 * vel_error_z  # Negative because NED

        # Attitude control
        tau_phi, tau_theta, tau_psi = self.update_attitude_control(
            attitude, angular_velocity, desired_roll, desired_pitch, dt
        )

        # Thrust with attitude compensation
        cos_correction = max(0.5, np.cos(phi) * np.cos(theta))
        thrust = (self.hover_thrust + thrust_adj) / cos_correction
        thrust = np.clip(thrust, 0, 2 * self.hover_thrust)

        return {
            "thrust": thrust,
            "tau_phi": tau_phi,
            "tau_theta": tau_theta,
            "tau_psi": tau_psi,
        }

    def get_gains_dict(self) -> Dict[str, Dict[str, float]]:
        """Return all PID gains as a dictionary for display/saving."""
        return {
            "position_x": {
                "Kp": self.pos_x_pid.gains.Kp,
                "Ki": self.pos_x_pid.gains.Ki,
                "Kd": self.pos_x_pid.gains.Kd,
            },
            "position_y": {
                "Kp": self.pos_y_pid.gains.Kp,
                "Ki": self.pos_y_pid.gains.Ki,
                "Kd": self.pos_y_pid.gains.Kd,
            },
            "position_z": {
                "Kp": self.pos_z_pid.gains.Kp,
                "Ki": self.pos_z_pid.gains.Ki,
                "Kd": self.pos_z_pid.gains.Kd,
            },
            "roll": {
                "Kp": self.roll_pid.gains.Kp,
                "Ki": self.roll_pid.gains.Ki,
                "Kd": self.roll_pid.gains.Kd,
            },
            "pitch": {
                "Kp": self.pitch_pid.gains.Kp,
                "Ki": self.pitch_pid.gains.Ki,
                "Kd": self.pitch_pid.gains.Kd,
            },
            "yaw": {
                "Kp": self.yaw_pid.gains.Kp,
                "Ki": self.yaw_pid.gains.Ki,
                "Kd": self.yaw_pid.gains.Kd,
            },
        }

    def tune_gains(
        self,
        controller_name: str,
        Kp: Optional[float] = None,
        Ki: Optional[float] = None,
        Kd: Optional[float] = None,
    ):
        """
        Tune gains for a specific controller at runtime.

        Args:
            controller_name: One of 'pos_x', 'pos_y', 'pos_z',
                           'roll', 'pitch', 'yaw'
            Kp, Ki, Kd: New gain values (None = keep current)
        """
        controller_map = {
            "pos_x": self.pos_x_pid,
            "pos_y": self.pos_y_pid,
            "pos_z": self.pos_z_pid,
            "roll": self.roll_pid,
            "pitch": self.pitch_pid,
            "yaw": self.yaw_pid,
            "roll_rate": self.roll_rate_pid,
            "pitch_rate": self.pitch_rate_pid,
            "yaw_rate": self.yaw_rate_pid,
        }

        if controller_name in controller_map:
            controller_map[controller_name].set_gains(Kp, Ki, Kd)


class RateLimiter:
    """
    Rate limiter for smooth control transitions.

    Limits the rate of change of a signal to prevent sudden jerks.
    """

    def __init__(self, rate_limit: float, initial_value: float = 0.0):
        """
        Args:
            rate_limit: Maximum rate of change per second
            initial_value: Starting value
        """
        self.rate_limit = rate_limit
        self.value = initial_value

    def update(self, target: float, dt: float) -> float:
        """
        Update the rate-limited value.

        Args:
            target: Desired value
            dt: Time step

        Returns:
            Rate-limited value
        """
        max_change = self.rate_limit * dt
        change = target - self.value

        if abs(change) > max_change:
            change = np.sign(change) * max_change

        self.value += change
        return self.value

    def reset(self, value: float = 0.0):
        """Reset to specified value."""
        self.value = value


if __name__ == "__main__":
    # Test PID controllers
    print("Testing PID Controller Module")
    print("=" * 50)

    # Test single PID controller
    gains = PIDGains(Kp=2.0, Ki=0.1, Kd=0.5)
    pid = PIDController(gains, "test")

    print("\nStep response test (setpoint = 10):")
    value = 0.0
    dt = 0.02

    for i in range(100):
        output = pid.update(10.0, value, dt)
        # Simple first-order system response
        value += output * dt
        if i % 20 == 0:
            print(f"  t={i*dt:.2f}s: value={value:.3f}, output={output:.3f}")

    print(f"\nFinal value: {value:.3f}")
    print(f"RMSE: {pid.get_rmse():.3f}")

    # Test quadcopter controller
    print("\n" + "=" * 50)
    print("Testing Quadcopter PID Controller")

    quad_pid = QuadcopterPIDController(mass=1.0, gravity=9.81)
    quad_pid.set_position_setpoint(5.0, 0.0, -3.0)  # 5m North, 3m altitude

    # Simulate control loop
    position = np.array([0.0, 0.0, 0.0])
    velocity = np.array([0.0, 0.0, 0.0])
    attitude = np.array([0.0, 0.0, 0.0])
    angular_velocity = np.array([0.0, 0.0, 0.0])

    print("\nSimulating position control...")
    for i in range(50):
        control = quad_pid.compute_control(
            position, velocity, attitude, angular_velocity, dt
        )

        # Simple dynamics update (very simplified)
        accel = np.array(
            [
                control["tau_theta"] * 2,  # pitch -> x acceleration
                -control["tau_phi"] * 2,  # roll -> y acceleration
                -(control["thrust"] - 9.81) / 1.0,  # thrust -> z acceleration
            ]
        )
        velocity += accel * dt
        position += velocity * dt

        if i % 10 == 0:
            print(
                f"  t={i*dt:.2f}s: pos=({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
            )

    print("\nGains dictionary:")
    for name, gains in quad_pid.get_gains_dict().items():
        print(f"  {name}: Kp={gains['Kp']}, Ki={gains['Ki']}, Kd={gains['Kd']}")

    print("\nPID Controller test complete!")
