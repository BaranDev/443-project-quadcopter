"""
visualization.py - Real-Time Visualization GUI for Quadcopter Simulator
CMSE443 Real-Time Systems Design - Term Project

This module provides a graphical user interface for:
- Real-time pose visualization (position, orientation)
- Telemetry display (last 10 position/velocity values)
- System parameter display and adjustment
- Control status and input visualization
- Start/Stop/Reset functionality

The GUI is built using tkinter for cross-platform compatibility
and uses a separate update thread to maintain ~20Hz refresh rate
without blocking the control loop.

References:
- Tkinter documentation
- Real-time GUI design patterns
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import math
import queue
from dataclasses import dataclass
from typing import Optional, Callable, Dict, List, Tuple, Any
from collections import deque


@dataclass
class TelemetryData:
    """Container for telemetry data to display."""

    # Position (NED frame)
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # Velocity
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    # Attitude (Euler angles in degrees for display)
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    # Angular velocity
    p: float = 0.0
    q: float = 0.0
    r: float = 0.0

    # Control status
    armed: bool = False
    connected: bool = False
    input_source: str = "None"
    control_mode: str = "Velocity"

    # Timing
    loop_rate_hz: float = 0.0
    loop_time_ms: float = 0.0
    deadline_misses: int = 0

    # Sensor data
    altitude_barometer: float = 0.0
    gps_lat: float = 0.0
    gps_lon: float = 0.0

    # Controller outputs
    thrust: float = 0.0
    roll_cmd: float = 0.0
    pitch_cmd: float = 0.0
    yaw_cmd: float = 0.0


class AttitudeIndicator(tk.Canvas):
    """
    Artificial horizon / attitude indicator widget.

    Displays roll and pitch angles graphically like an aircraft
    attitude indicator instrument.
    """

    def __init__(self, parent, size: int = 150, **kwargs):
        super().__init__(
            parent, width=size, height=size, bg="black", highlightthickness=1, **kwargs
        )
        self.size = size
        self.center = size // 2
        self.roll = 0.0  # degrees
        self.pitch = 0.0  # degrees

        # Colors
        self.sky_color = "#0066CC"
        self.ground_color = "#664400"
        self.horizon_color = "white"

        self._draw()

    def _draw(self):
        """Draw the attitude indicator with proper roll rotation."""
        self.delete("all")

        center = self.center
        radius = self.size // 2 - 10

        # Convert angles
        roll_rad = math.radians(self.roll)
        pitch_offset = self.pitch * 2  # Scale pitch to pixels

        # Calculate rotated horizon line endpoints
        # The horizon rotates around center based on roll
        # and shifts up/down based on pitch
        horizon_length = self.size * 2  # Long enough to cover corners when rotated
        
        # Horizon center point (shifted by pitch)
        hx = center
        hy = center + pitch_offset
        
        # Calculate rotated endpoints
        dx = horizon_length / 2 * math.cos(roll_rad)
        dy = horizon_length / 2 * math.sin(roll_rad)
        
        x1, y1 = hx - dx, hy - dy
        x2, y2 = hx + dx, hy + dy
        
        # Create clipping region (circular)
        # Draw sky and ground as polygons that rotate with the horizon
        
        # Sky polygon (above the tilted horizon)
        sky_points = [
            x1, y1,
            x2, y2,
            x2 - horizon_length * math.sin(roll_rad), y2 - horizon_length * math.cos(roll_rad),
            x1 - horizon_length * math.sin(roll_rad), y1 - horizon_length * math.cos(roll_rad),
        ]
        self.create_polygon(*sky_points, fill=self.sky_color, outline="")
        
        # Ground polygon (below the tilted horizon)
        ground_points = [
            x1, y1,
            x2, y2,
            x2 + horizon_length * math.sin(roll_rad), y2 + horizon_length * math.cos(roll_rad),
            x1 + horizon_length * math.sin(roll_rad), y1 + horizon_length * math.cos(roll_rad),
        ]
        self.create_polygon(*ground_points, fill=self.ground_color, outline="")

        # Horizon line (tilted)
        self.create_line(x1, y1, x2, y2, fill=self.horizon_color, width=2)

        # Draw circular mask/border to hide overflow
        self.create_oval(
            center - radius - 2, center - radius - 2,
            center + radius + 2, center + radius + 2,
            outline="#1a1a2e", width=20
        )
        self.create_oval(
            center - radius, center - radius,
            center + radius, center + radius,
            outline="white", width=2
        )

        # Roll indicator arc at top
        arc_radius = radius - 5
        self.create_arc(
            center - arc_radius,
            center - arc_radius,
            center + arc_radius,
            center + arc_radius,
            start=210,
            extent=120,
            style="arc",
            outline="white",
            width=1,
        )

        # Roll indicator triangle (fixed at top, shows current roll)
        indicator_angle = -roll_rad  # Opposite direction to show bank
        tri_x = center + arc_radius * math.sin(indicator_angle)
        tri_y = center - arc_radius * math.cos(indicator_angle)
        self.create_polygon(
            center, center - arc_radius + 8,
            center - 5, center - arc_radius,
            center + 5, center - arc_radius,
            fill="yellow",
            outline="white",
        )

        # Aircraft symbol (fixed reference at center)
        self.create_line(
            center - 30, center, center - 10, center, fill="yellow", width=3
        )
        self.create_line(
            center + 10, center, center + 30, center, fill="yellow", width=3
        )
        self.create_oval(
            center - 5,
            center - 5,
            center + 5,
            center + 5,
            fill="yellow",
            outline="yellow",
        )

        # Pitch ladder lines (rotated with horizon)
        for pitch_mark in [-20, -10, 10, 20]:
            # Calculate position relative to horizon center
            mark_y_offset = pitch_offset - pitch_mark * 2
            
            # Rotate the line endpoints based on roll
            line_half_width = 15 if abs(pitch_mark) == 10 else 25
            
            # Center of this pitch mark
            mark_cx = center
            mark_cy = center + mark_y_offset
            
            # Calculate rotated endpoints
            ldx = line_half_width * math.cos(roll_rad)
            ldy = line_half_width * math.sin(roll_rad)
            
            lx1, ly1 = mark_cx - ldx, mark_cy - ldy
            lx2, ly2 = mark_cx + ldx, mark_cy + ldy
            
            # Only draw if within visible area
            if 25 < mark_cy < self.size - 25:
                self.create_line(lx1, ly1, lx2, ly2, fill="white", width=1)
                # Add pitch value text
                self.create_text(
                    lx2 + 8, ly2,
                    text=str(pitch_mark),
                    fill="white",
                    font=("Arial", 7),
                )

        # Angle displays
        self.create_text(
            10,
            self.size - 10,
            anchor="sw",
            text=f"R:{self.roll:+.1f}°",
            fill="white",
            font=("Consolas", 8),
        )
        self.create_text(
            self.size - 10,
            self.size - 10,
            anchor="se",
            text=f"P:{self.pitch:+.1f}°",
            fill="white",
            font=("Consolas", 8),
        )

    def update_attitude(self, roll: float, pitch: float):
        """Update displayed attitude (degrees)."""
        self.roll = max(-90, min(90, roll))
        self.pitch = max(-90, min(90, pitch))
        self._draw()


class HeadingIndicator(tk.Canvas):
    """
    Compass / heading indicator widget.

    Displays yaw angle as a compass heading.
    """

    def __init__(self, parent, size: int = 150, **kwargs):
        super().__init__(
            parent, width=size, height=size, bg="black", highlightthickness=1, **kwargs
        )
        self.size = size
        self.center = size // 2
        self.heading = 0.0  # degrees

        self._draw()

    def _draw(self):
        """Draw the heading indicator."""
        self.delete("all")

        center = self.center
        radius = self.size // 2 - 15

        # Draw compass rose
        self.create_oval(
            center - radius,
            center - radius,
            center + radius,
            center + radius,
            outline="white",
            width=2,
        )

        # Draw cardinal directions
        heading_rad = math.radians(self.heading)

        cardinals = [("N", 0), ("E", 90), ("S", 180), ("W", 270)]
        for label, angle in cardinals:
            angle_rad = math.radians(angle - self.heading)
            x = center + (radius - 12) * math.sin(angle_rad)
            y = center - (radius - 12) * math.cos(angle_rad)
            color = "red" if label == "N" else "white"
            self.create_text(x, y, text=label, fill=color, font=("Arial", 10, "bold"))

        # Draw tick marks
        for i in range(36):
            angle = i * 10 - self.heading
            angle_rad = math.radians(angle)

            if i % 9 == 0:  # Major tick (cardinal)
                inner = radius - 20
            elif i % 3 == 0:  # Medium tick
                inner = radius - 12
            else:  # Minor tick
                inner = radius - 7

            x1 = center + inner * math.sin(angle_rad)
            y1 = center - inner * math.cos(angle_rad)
            x2 = center + radius * math.sin(angle_rad)
            y2 = center - radius * math.cos(angle_rad)

            self.create_line(x1, y1, x2, y2, fill="white", width=1)

        # Aircraft symbol (fixed at center pointing up)
        self.create_polygon(
            center,
            center - 20,
            center - 8,
            center + 10,
            center,
            center + 5,
            center + 8,
            center + 10,
            fill="yellow",
            outline="white",
        )

        # Heading display
        self.create_text(
            center,
            self.size - 8,
            text=f"HDG: {self.heading:.0f}°",
            fill="white",
            font=("Consolas", 9),
        )

    def update_heading(self, heading: float):
        """Update displayed heading (degrees, 0-360)."""
        self.heading = heading % 360
        self._draw()


class PositionPlot(tk.Canvas):
    """
    Top-down position plot showing drone trajectory.

    Displays:
    - Current position
    - Last 10 positions (trajectory)
    - North/East axes
    """

    def __init__(self, parent, size: int = 200, **kwargs):
        super().__init__(
            parent,
            width=size,
            height=size,
            bg="#1a1a2e",
            highlightthickness=1,
            **kwargs,
        )
        self.size = size
        self.center = size // 2

        # Position history (last 10)
        self.positions: deque = deque(maxlen=10)
        self.current_x = 0.0
        self.current_y = 0.0
        self.yaw = 0.0

        # Scale (meters per pixel)
        self.meters_per_pixel = 20.0 / (size // 2 - 20)  # 20m range

        self._draw()

    def _world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        screen_x = self.center + int(y / self.meters_per_pixel)  # East -> right
        screen_y = self.center - int(x / self.meters_per_pixel)  # North -> up
        return screen_x, screen_y

    def _draw(self):
        """Draw the position plot."""
        self.delete("all")

        center = self.center

        # Grid
        for i in range(-2, 3):
            # Vertical lines (East-West)
            offset = i * 40
            self.create_line(
                center + offset,
                10,
                center + offset,
                self.size - 10,
                fill="#333355",
                width=1,
            )
            # Horizontal lines (North-South)
            self.create_line(
                10,
                center + offset,
                self.size - 10,
                center + offset,
                fill="#333355",
                width=1,
            )

        # Axes
        self.create_line(
            center, 10, center, self.size - 10, fill="#666688", width=1
        )  # N-S axis
        self.create_line(
            10, center, self.size - 10, center, fill="#666688", width=1
        )  # E-W axis

        # Axis labels
        self.create_text(center, 8, text="N", fill="red", font=("Arial", 8))
        self.create_text(
            center, self.size - 8, text="S", fill="white", font=("Arial", 8)
        )
        self.create_text(8, center, text="W", fill="white", font=("Arial", 8))
        self.create_text(
            self.size - 8, center, text="E", fill="white", font=("Arial", 8)
        )

        # Draw trajectory (last 10 positions)
        if len(self.positions) > 1:
            points = []
            for x, y in self.positions:
                sx, sy = self._world_to_screen(x, y)
                points.extend([sx, sy])

            self.create_line(*points, fill="#00ff88", width=2, smooth=True)

        # Draw position markers
        for i, (x, y) in enumerate(self.positions):
            sx, sy = self._world_to_screen(x, y)
            alpha = (i + 1) / len(self.positions) if self.positions else 1
            radius = 2 + int(alpha * 3)
            self.create_oval(
                sx - radius,
                sy - radius,
                sx + radius,
                sy + radius,
                fill="#00ff88",
                outline="",
            )

        # Draw current position with direction
        sx, sy = self._world_to_screen(self.current_x, self.current_y)

        # Drone triangle (pointing in yaw direction)
        yaw_rad = math.radians(self.yaw)
        size = 12
        points = [
            sx + size * math.sin(yaw_rad),
            sy - size * math.cos(yaw_rad),
            sx + size * 0.5 * math.sin(yaw_rad + 2.5),
            sy - size * 0.5 * math.cos(yaw_rad + 2.5),
            sx + size * 0.5 * math.sin(yaw_rad - 2.5),
            sy - size * 0.5 * math.cos(yaw_rad - 2.5),
        ]
        self.create_polygon(*points, fill="yellow", outline="white")

        # Position text
        self.create_text(
            5,
            5,
            anchor="nw",
            text=f"X:{self.current_x:+.1f}m",
            fill="white",
            font=("Consolas", 8),
        )
        self.create_text(
            5,
            18,
            anchor="nw",
            text=f"Y:{self.current_y:+.1f}m",
            fill="white",
            font=("Consolas", 8),
        )

    def update_position(self, x: float, y: float, yaw: float):
        """Update displayed position."""
        self.positions.append((x, y))
        self.current_x = x
        self.current_y = y
        self.yaw = yaw
        self._draw()

    def clear_history(self):
        """Clear position history."""
        self.positions.clear()
        self._draw()


class VisualizationGUI:
    """
    Main visualization GUI for the quadcopter simulator.

    Features:
    - Real-time telemetry display (~20Hz update)
    - Attitude indicator (artificial horizon)
    - Heading indicator (compass)
    - Position plot (top-down view)
    - Last 10 position/velocity values table
    - System parameters display
    - Start/Stop/Reset controls
    - PID gain adjustment

    The GUI runs in a separate thread and communicates with the
    main control loop via thread-safe queues.
    """

    def __init__(self, title: str = "CMSE443 Quadcopter Simulator"):
        """Initialize the GUI."""
        self.title = title
        self.root: Optional[tk.Tk] = None

        # Thread-safe communication
        self.telemetry_queue: queue.Queue = queue.Queue(maxsize=10)
        self.command_queue: queue.Queue = queue.Queue()

        # Callbacks
        self._on_start: Optional[Callable] = None
        self._on_stop: Optional[Callable] = None
        self._on_reset: Optional[Callable] = None
        self._on_arm: Optional[Callable] = None
        self._on_parameter_change: Optional[Callable] = None

        # State
        self.running = False
        self.gui_thread: Optional[threading.Thread] = None

        # Data storage
        self.position_history: List[Tuple[float, float, float]] = []
        self.velocity_history: List[Tuple[float, float, float]] = []

        # GUI elements (initialized in _create_gui)
        self.attitude_indicator: Optional[AttitudeIndicator] = None
        self.heading_indicator: Optional[HeadingIndicator] = None
        self.position_plot: Optional[PositionPlot] = None
        self.root: Optional[tk.Tk] = None
        self.history_tree: Optional[ttk.Treeview] = None
        self.status_label: Optional[ttk.Label] = None

    def set_callbacks(
        self,
        on_start: Optional[Callable[[], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
        on_reset: Optional[Callable[[], None]] = None,
        on_arm: Optional[Callable[[bool], None]] = None,
        on_parameter_change: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """Set callback functions for GUI events."""
        self._on_start = on_start
        self._on_stop = on_stop
        self._on_reset = on_reset
        self._on_arm = on_arm
        self._on_parameter_change = on_parameter_change

    def _create_gui(self):
        """Create all GUI elements."""
        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.configure(bg="#1a1a2e")
        self.root.geometry("1000x700")
        self.root.minsize(900, 600)

        # Configure styles
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1a1a2e")
        style.configure("TLabel", background="#1a1a2e", foreground="white")
        style.configure("TButton", background="#333355")
        style.configure("Header.TLabel", font=("Arial", 12, "bold"))
        style.configure("Data.TLabel", font=("Consolas", 10))

        # Main container
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel - Instruments
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        self._create_instrument_panel(left_frame)

        # Center panel - Telemetry
        center_frame = ttk.Frame(main_frame)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self._create_telemetry_panel(center_frame)

        # Right panel - Controls and Parameters
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self._create_control_panel(right_frame)
        self._create_parameter_panel(right_frame)

        # Status bar
        self._create_status_bar()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _create_instrument_panel(self, parent):
        """Create the instrument panel with attitude and heading indicators."""
        ttk.Label(parent, text="INSTRUMENTS", style="Header.TLabel").pack(pady=(0, 10))

        # Attitude indicator
        att_frame = ttk.LabelFrame(parent, text="Attitude")
        att_frame.pack(fill=tk.X, pady=5)

        self.attitude_indicator = AttitudeIndicator(att_frame, size=180)
        self.attitude_indicator.pack(padx=5, pady=5)

        # Heading indicator
        hdg_frame = ttk.LabelFrame(parent, text="Heading")
        hdg_frame.pack(fill=tk.X, pady=5)

        self.heading_indicator = HeadingIndicator(hdg_frame, size=180)
        self.heading_indicator.pack(padx=5, pady=5)

        # Position plot
        pos_frame = ttk.LabelFrame(parent, text="Position (Top View)")
        pos_frame.pack(fill=tk.X, pady=5)

        self.position_plot = PositionPlot(pos_frame, size=180)
        self.position_plot.pack(padx=5, pady=5)

    def _create_telemetry_panel(self, parent):
        """Create the telemetry display panel."""
        ttk.Label(parent, text="TELEMETRY", style="Header.TLabel").pack(pady=(0, 10))

        # Current values frame
        current_frame = ttk.LabelFrame(parent, text="Current State")
        current_frame.pack(fill=tk.X, pady=5)

        # Create labels for current values
        self.current_labels = {}

        row = 0
        for category, items in [
            ("Position", [("X", "m"), ("Y", "m"), ("Z", "m")]),
            ("Velocity", [("Vx", "m/s"), ("Vy", "m/s"), ("Vz", "m/s")]),
            ("Attitude", [("Roll", "°"), ("Pitch", "°"), ("Yaw", "°")]),
            ("Angular Rate", [("P", "°/s"), ("Q", "°/s"), ("R", "°/s")]),
        ]:
            ttk.Label(
                current_frame, text=category + ":", font=("Arial", 9, "bold")
            ).grid(row=row, column=0, sticky="w", padx=5)

            col = 1
            for name, unit in items:
                key = name.lower()
                lbl = ttk.Label(
                    current_frame,
                    text=f"{name}: 0.00{unit}",
                    style="Data.TLabel",
                    width=15,
                )
                lbl.grid(row=row, column=col, padx=2)
                self.current_labels[key] = lbl
                col += 1
            row += 1

        # History table (last 10 values) - fixed size
        history_frame = ttk.LabelFrame(parent, text="Position History (Last 10)")
        history_frame.pack(fill=tk.X, pady=5)

        # Create treeview for history (fixed 10 rows)
        columns = ("idx", "x", "y", "z", "vx", "vy", "vz")
        self.history_tree = ttk.Treeview(
            history_frame, columns=columns, show="headings", height=10
        )

        self.history_tree.heading("idx", text="#")
        self.history_tree.heading("x", text="X")
        self.history_tree.heading("y", text="Y")
        self.history_tree.heading("z", text="Z")
        self.history_tree.heading("vx", text="Vx")
        self.history_tree.heading("vy", text="Vy")
        self.history_tree.heading("vz", text="Vz")

        for col in columns:
            self.history_tree.column(col, width=50, anchor="center")
        self.history_tree.column("idx", width=25)

        self.history_tree.pack(fill=tk.X)

        # Real-time metrics
        metrics_frame = ttk.LabelFrame(parent, text="Real-Time Metrics")
        metrics_frame.pack(fill=tk.X, pady=5)

        self.metrics_labels = {}
        metrics = [
            ("loop_rate", "Loop Rate:", "Hz"),
            ("loop_time", "Loop Time:", "ms"),
            ("deadline_misses", "Deadline Misses:", ""),
            ("input_source", "Input Source:", ""),
        ]

        for i, (key, label, unit) in enumerate(metrics):
            ttk.Label(metrics_frame, text=label).grid(
                row=i // 2, column=(i % 2) * 2, sticky="w", padx=5, pady=2
            )
            lbl = ttk.Label(metrics_frame, text=f"0 {unit}", style="Data.TLabel")
            lbl.grid(row=i // 2, column=(i % 2) * 2 + 1, sticky="w", padx=5, pady=2)
            self.metrics_labels[key] = lbl

    def _create_control_panel(self, parent):
        """Create the control buttons panel."""
        ttk.Label(parent, text="CONTROLS", style="Header.TLabel").pack(pady=(0, 10))

        control_frame = ttk.LabelFrame(parent, text="Simulation Control")
        control_frame.pack(fill=tk.X, pady=5)

        # Start/Stop/Reset buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        self.start_btn = ttk.Button(
            btn_frame, text="▶ Start", command=self._on_start_click
        )
        self.start_btn.pack(side=tk.LEFT, padx=2)

        self.stop_btn = ttk.Button(
            btn_frame, text="⏹ Stop", command=self._on_stop_click, state="disabled"
        )
        self.stop_btn.pack(side=tk.LEFT, padx=2)

        self.reset_btn = ttk.Button(
            btn_frame, text="↺ Reset", command=self._on_reset_click
        )
        self.reset_btn.pack(side=tk.LEFT, padx=2)

        # Arm/Disarm
        arm_frame = ttk.Frame(control_frame)
        arm_frame.pack(fill=tk.X, padx=5, pady=5)

        self.arm_var = tk.BooleanVar(value=False)
        self.arm_check = ttk.Checkbutton(
            arm_frame, text="Armed", variable=self.arm_var, command=self._on_arm_click
        )
        self.arm_check.pack(side=tk.LEFT)

        self.arm_status = ttk.Label(
            arm_frame, text="DISARMED", foreground="red", font=("Arial", 10, "bold")
        )
        self.arm_status.pack(side=tk.LEFT, padx=10)

        # PID Toggle
        pid_toggle_frame = ttk.Frame(control_frame)
        pid_toggle_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.pid_enabled = tk.BooleanVar(value=True)
        self.pid_check = ttk.Checkbutton(
            pid_toggle_frame, 
            text="Enable PID Controller", 
            variable=self.pid_enabled,
            command=self._on_apply_params  # Update immediately
        )
        self.pid_check.pack(side=tk.LEFT)
        
        # Hover Toggle
        hover_toggle_frame = ttk.Frame(control_frame)
        hover_toggle_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.hover_enabled = tk.BooleanVar(value=False)
        self.hover_check = ttk.Checkbutton(
            hover_toggle_frame, 
            text="Enable Hover (hold position)", 
            variable=self.hover_enabled,
            command=self._on_hover_toggle
        )
        self.hover_check.pack(side=tk.LEFT)
        
        self.hover_status = ttk.Label(
            hover_toggle_frame, text="OFF", foreground="gray", font=("Arial", 9, "bold")
        )
        self.hover_status.pack(side=tk.LEFT, padx=10)

        # Connection status
        conn_frame = ttk.Frame(control_frame)
        conn_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(conn_frame, text="Connection:").pack(side=tk.LEFT)
        self.conn_status = ttk.Label(conn_frame, text="Disconnected", foreground="red")
        self.conn_status.pack(side=tk.LEFT, padx=5)

        # Control mode selection
        mode_frame = ttk.LabelFrame(parent, text="Control Mode")
        mode_frame.pack(fill=tk.X, pady=5)

        self.control_mode = tk.StringVar(value="velocity")
        modes = [
            ("Velocity", "velocity"),
            ("Position", "position"),
            ("Attitude", "attitude"),
        ]

        for text, value in modes:
            rb = ttk.Radiobutton(
                mode_frame, text=text, value=value, variable=self.control_mode
            )
            rb.pack(anchor="w", padx=5)
            
        # Integration Method selection
        int_frame = ttk.LabelFrame(parent, text="Integration Method")
        int_frame.pack(fill=tk.X, pady=5)

        self.int_method = tk.StringVar(value="euler")
        int_methods = [
            ("Euler", "euler"),
            ("Runge-Kutta 4", "rk4"),
        ]

        for text, value in int_methods:
            rb = ttk.Radiobutton(
                int_frame, 
                text=text, 
                value=value, 
                variable=self.int_method,
                command=self._on_apply_params
            )
            rb.pack(anchor="w", padx=5)

    def _create_parameter_panel(self, parent):
        """Create the parameter adjustment panel."""
        param_frame = ttk.LabelFrame(parent, text="Parameters")
        param_frame.pack(fill=tk.X, pady=5)

        # System parameters
        self.param_entries = {}

        params = [
            ("mass", "Mass (kg):", 1.0, 0.1, 10.0),
            ("max_velocity", "Max Vel (m/s):", 5.0, 1.0, 20.0),
            ("max_yaw_rate", "Max Yaw (°/s):", 60.0, 10.0, 180.0),
        ]

        for i, (key, label, default, min_val, max_val) in enumerate(params):
            ttk.Label(param_frame, text=label).grid(
                row=i, column=0, sticky="w", padx=5, pady=2
            )

            var = tk.DoubleVar(value=default)
            spin = ttk.Spinbox(
                param_frame,
                from_=min_val,
                to=max_val,
                textvariable=var,
                width=8,
                increment=0.1,
            )
            spin.grid(row=i, column=1, padx=5, pady=2)
            self.param_entries[key] = var

        # PID Gains (simplified)
        pid_frame = ttk.LabelFrame(parent, text="PID Gains")
        pid_frame.pack(fill=tk.X, pady=5)

        self.pid_entries = {}

        pid_params = [
            ("pos_Kp", "Position Kp:", 2.0),
            ("pos_Kd", "Position Kd:", 1.5),
            ("att_Kp", "Attitude Kp:", 6.0),
            ("att_Kd", "Attitude Kd:", 1.2),
        ]

        for i, (key, label, default) in enumerate(pid_params):
            ttk.Label(pid_frame, text=label).grid(
                row=i, column=0, sticky="w", padx=5, pady=2
            )

            var = tk.DoubleVar(value=default)
            spin = ttk.Spinbox(
                pid_frame, from_=0, to=20, textvariable=var, width=8, increment=0.1
            )
            spin.grid(row=i, column=1, padx=5, pady=2)
            self.pid_entries[key] = var

        # Apply button
        ttk.Button(
            param_frame, text="Apply Parameters", command=self._on_apply_params
        ).grid(row=len(params), column=0, columnspan=2, pady=5)

    def _create_status_bar(self):
        """Create the status bar at the bottom."""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_label = ttk.Label(
            status_frame,
            text="Ready | Press Start to begin simulation",
            relief=tk.SUNKEN,
        )
        self.status_label.pack(fill=tk.X, padx=2, pady=2)

    def _on_start_click(self):
        """Handle Start button click."""
        if self.start_btn:
            self.start_btn.configure(state="disabled")
        if self.stop_btn:
            self.stop_btn.configure(state="normal")
        if self.status_label:
            self.status_label.configure(text="Running...")
        if self._on_start:
            self._on_start()

    def _on_stop_click(self):
        """Handle Stop button click."""
        if self.start_btn:
            self.start_btn.configure(state="normal")
        if self.stop_btn:
            self.stop_btn.configure(state="disabled")
        if self.status_label:
            self.status_label.configure(text="Stopped")
        if self._on_stop:
            self._on_stop()

    def _on_reset_click(self):
        """Handle Reset button click."""
        if self.position_plot:
            self.position_plot.clear_history()
        if self.arm_var:
            self.arm_var.set(False)
        if self.arm_status:
            self.arm_status.configure(text="DISARMED", foreground="red")

        # Clear history table
        if self.history_tree:
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)

        if self.status_label:
            self.status_label.configure(text="Reset complete")
        if self._on_reset:
            self._on_reset()

    def _on_arm_click(self):
        """Handle Arm checkbox click."""
        armed = False
        if self.arm_var:
            armed = self.arm_var.get()
            if self.arm_status:
                if armed:
                    self.arm_status.configure(text="ARMED", foreground="green")
                else:
                    self.arm_status.configure(text="DISARMED", foreground="red")
        if self._on_arm:
            self._on_arm(armed)

    def _on_hover_toggle(self):
        """Handle Hover checkbox toggle."""
        enabled = self.hover_enabled.get()
        if self.hover_status:
            if enabled:
                self.hover_status.configure(text="ON", foreground="green")
            else:
                self.hover_status.configure(text="OFF", foreground="gray")
        self._on_apply_params()

    def _on_apply_params(self):
        """Handle Apply Parameters button click."""
        params = {}
        for key, var in self.param_entries.items():
            params[key] = var.get()
        for key, var in self.pid_entries.items():
            params[key] = var.get()
            
        # Add new parameters
        if hasattr(self, "pid_enabled"):
            params["pid_enabled"] = self.pid_enabled.get()
        if hasattr(self, "int_method"):
            params["integration_method"] = self.int_method.get()
        if hasattr(self, "hover_enabled"):
            params["hover_enabled"] = self.hover_enabled.get()

        if self._on_parameter_change:
            self._on_parameter_change(params)

        if self.status_label:
            self.status_label.configure(text="Parameters applied")

    def _on_close(self):
        """Handle window close."""
        self.running = False
        if self.root:
            self.root.quit()
            self.root.destroy()

    def _update_gui(self):
        """Update GUI with latest telemetry (called from main thread)."""
        try:
            # Process all available telemetry updates
            while not self.telemetry_queue.empty():
                data: TelemetryData = self.telemetry_queue.get_nowait()

                # Update current value labels
                self.current_labels["x"].configure(text=f"X: {data.x:+.2f}m")
                self.current_labels["y"].configure(text=f"Y: {data.y:+.2f}m")
                self.current_labels["z"].configure(text=f"Z: {data.z:+.2f}m")
                self.current_labels["vx"].configure(text=f"Vx: {data.vx:+.2f}m/s")
                self.current_labels["vy"].configure(text=f"Vy: {data.vy:+.2f}m/s")
                self.current_labels["vz"].configure(text=f"Vz: {data.vz:+.2f}m/s")
                self.current_labels["roll"].configure(text=f"Roll: {data.roll:+.1f}°")
                self.current_labels["pitch"].configure(
                    text=f"Pitch: {data.pitch:+.1f}°"
                )
                self.current_labels["yaw"].configure(text=f"Yaw: {data.yaw:+.1f}°")
                self.current_labels["p"].configure(text=f"P: {data.p:+.1f}°/s")
                self.current_labels["q"].configure(text=f"Q: {data.q:+.1f}°/s")
                self.current_labels["r"].configure(text=f"R: {data.r:+.1f}°/s")

                # Update metrics
                self.metrics_labels["loop_rate"].configure(
                    text=f"{data.loop_rate_hz:.1f} Hz"
                )
                self.metrics_labels["loop_time"].configure(
                    text=f"{data.loop_time_ms:.2f} ms"
                )
                self.metrics_labels["deadline_misses"].configure(
                    text=str(data.deadline_misses)
                )
                self.metrics_labels["input_source"].configure(text=data.input_source)

                # Update instruments (guard against None)
                if self.attitude_indicator:
                    self.attitude_indicator.update_attitude(data.roll, data.pitch)
                if self.heading_indicator:
                    self.heading_indicator.update_heading(data.yaw)
                if self.position_plot:
                    self.position_plot.update_position(data.x, data.y, data.yaw)

                # Update connection status
                if data.connected:
                    self.conn_status.configure(text="Connected", foreground="green")
                else:
                    self.conn_status.configure(text="Disconnected", foreground="red")

                # Update arm status
                if data.armed:
                    self.arm_var.set(True)
                    self.arm_status.configure(text="ARMED", foreground="green")
                else:
                    self.arm_var.set(False)
                    self.arm_status.configure(text="DISARMED", foreground="red")
                    
                # Update position/velocity history for the table
                self.position_history.append((data.x, data.y, data.z))
                self.velocity_history.append((data.vx, data.vy, data.vz))
                # Keep only last 10
                if len(self.position_history) > 10:
                    self.position_history.pop(0)
                if len(self.velocity_history) > 10:
                    self.velocity_history.pop(0)
                    
                # Update history table
                if self.history_tree:
                    # Clear existing
                    for item in self.history_tree.get_children():
                        self.history_tree.delete(item)
                    # Add new data
                    for i, (pos, vel) in enumerate(zip(self.position_history, self.velocity_history)):
                        values = (
                            i + 1,
                            f"{pos[0]:.2f}",
                            f"{pos[1]:.2f}",
                            f"{pos[2]:.2f}",
                            f"{vel[0]:.2f}",
                            f"{vel[1]:.2f}",
                            f"{vel[2]:.2f}",
                        )
                        self.history_tree.insert("", "end", values=values)

        except queue.Empty:
            pass

        # Schedule next update (~20Hz)
        if self.running and self.root:
            self.root.after(50, self._update_gui)

    def update_telemetry(self, data: TelemetryData):
        """
        Update telemetry display (thread-safe).

        Called from the control loop thread.
        """
        try:
            self.telemetry_queue.put_nowait(data)
        except queue.Full:
            # Drop oldest data
            try:
                self.telemetry_queue.get_nowait()
                self.telemetry_queue.put_nowait(data)
            except:
                pass

    def update_history_table(self, positions: List, velocities: List):
        """Update the history table with new data."""
        if not self.history_tree:
            return

        # Clear existing
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)

        # Add new data
        for i, (pos, vel) in enumerate(zip(positions, velocities)):
            values = (
                i + 1,
                f"{pos[0]:.2f}",
                f"{pos[1]:.2f}",
                f"{pos[2]:.2f}",
                f"{vel[0]:.2f}",
                f"{vel[1]:.2f}",
                f"{vel[2]:.2f}",
            )
            self.history_tree.insert("", "end", values=values)

    def set_status(self, message: str):
        """Set status bar message (thread-safe)."""
        if self.root:

            def update_status():
                if self.status_label:
                    self.status_label.configure(text=message)

            self.root.after(0, update_status)

    def start(self):
        """Start the GUI in a separate thread."""
        self.running = True
        self.gui_thread = threading.Thread(target=self._run_gui, daemon=True)
        self.gui_thread.start()

    def start_in_main_thread(self):
        """Start the GUI in the main thread (required for Windows)."""
        self.running = True
        self._create_gui()
        if self.root:
            self.root.after(50, self._update_gui)
            # Don't call mainloop here - let caller handle it
    
    def run_mainloop(self):
        """Run the tkinter mainloop (call from main thread)."""
        if self.root:
            self.root.mainloop()

    def _run_gui(self):
        """Run the GUI main loop."""
        self._create_gui()
        if self.root:
            self.root.after(50, self._update_gui)
            self.root.mainloop()

    def stop(self):
        """Stop the GUI."""
        self.running = False
        if self.root:
            try:
                self.root.after(0, lambda: self._on_close())
            except Exception:
                pass

    def is_running(self) -> bool:
        """Check if GUI is running."""
        return self.running


if __name__ == "__main__":
    # Test the GUI
    import random

    print("Testing Visualization GUI")
    print("=" * 50)

    # Create and start GUI
    gui = VisualizationGUI("Test GUI")

    def on_start():
        print("Start clicked")

    def on_stop():
        print("Stop clicked")

    def on_reset():
        print("Reset clicked")

    def on_arm(armed):
        print(f"Arm changed: {armed}")

    gui.set_callbacks(
        on_start=on_start, on_stop=on_stop, on_reset=on_reset, on_arm=on_arm
    )

    gui.start()

    # Simulate telemetry updates
    print("\nSimulating telemetry for 30 seconds...")
    print("Close the window or wait to exit")

    t = 0
    try:
        while gui.is_running() and t < 30:
            # Generate test data
            data = TelemetryData(
                x=5 * math.sin(t * 0.5),
                y=5 * math.cos(t * 0.5),
                z=-3 + 0.5 * math.sin(t * 0.3),
                vx=2.5 * math.cos(t * 0.5),
                vy=-2.5 * math.sin(t * 0.5),
                vz=0.15 * math.cos(t * 0.3),
                roll=10 * math.sin(t * 0.8),
                pitch=8 * math.cos(t * 0.6),
                yaw=(t * 20) % 360,
                p=8 * math.cos(t * 0.8),
                q=-4.8 * math.sin(t * 0.6),
                r=20,
                armed=True,
                connected=True,
                input_source="Xbox Controller",
                loop_rate_hz=50 + random.uniform(-2, 2),
                loop_time_ms=20 + random.uniform(-1, 1),
                deadline_misses=0,
            )

            gui.update_telemetry(data)

            time.sleep(0.05)  # 20Hz update
            t += 0.05

    except KeyboardInterrupt:
        print("\nInterrupted")

    gui.stop()
    print("\nGUI test complete!")
