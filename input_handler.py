"""
input_handler.py - Unified Input Handler for Xbox Controller and Keyboard
CMSE443 Real-Time Systems Design - Term Project

This module provides a unified input abstraction layer that supports:
- Xbox Wireless Controller (via XInput API on Windows)
- Keyboard fallback (via pynput library)

The input handler implements:
- Dead zone filtering for analog sticks
- Input smoothing to reduce noise
- Configurable sensitivity curves
- Seamless fallback between input methods

Compatible Controllers:
- Xbox Wireless Controller (Series X|S, Xbox One)
- Windows 10/11 compatible Xbox controllers

Reference:
- Microsoft XInput API Documentation
- Real-time input polling best practices
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Tuple, Any
from enum import Enum, auto
import math

# Try to import XInput for Xbox controller support
XInput: Optional[Any] = None
XINPUT_AVAILABLE = False
try:
    import XInput as _XInput

    XInput = _XInput
    XINPUT_AVAILABLE = True
except ImportError:
    print("Warning: XInput not available. Install with: pip install XInput-Python")

# Try to import keyboard for keyboard input
keyboard: Optional[Any] = None
KEYBOARD_AVAILABLE = False
try:
    import keyboard as _keyboard

    keyboard = _keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    print("Warning: keyboard library not available. Install with: pip install keyboard")


class InputSource(Enum):
    """Input source types."""

    NONE = auto()
    XBOX_CONTROLLER = auto()
    KEYBOARD = auto()


@dataclass
class ControllerState:
    """
    Normalized controller state.

    All values are normalized to [-1, 1] or [0, 1] ranges.
    This provides a unified interface regardless of input source.
    """

    # Left stick (throttle/yaw)
    left_stick_x: float = 0.0  # Yaw: -1 (left) to +1 (right)
    left_stick_y: float = 0.0  # Throttle: -1 (down) to +1 (up)

    # Right stick (pitch/roll)
    right_stick_x: float = 0.0  # Roll: -1 (left) to +1 (right)
    right_stick_y: float = 0.0  # Pitch: -1 (down/back) to +1 (up/forward)

    # Triggers (altitude)
    left_trigger: float = 0.0  # Descend: 0 to 1
    right_trigger: float = 0.0  # Ascend: 0 to 1

    # Buttons
    button_a: bool = False  # Arm/Disarm toggle
    button_b: bool = False  # Emergency stop
    button_x: bool = False  # Reset simulation
    button_y: bool = False  # Toggle control mode
    button_start: bool = False  # Start
    button_back: bool = False  # Back/Select
    button_lb: bool = False  # Left bumper (decrease sensitivity)
    button_rb: bool = False  # Right bumper (increase sensitivity)
    dpad_up: bool = False  # D-pad up
    dpad_down: bool = False  # D-pad down
    dpad_left: bool = False  # D-pad left
    dpad_right: bool = False  # D-pad right

    # Metadata
    source: InputSource = InputSource.NONE
    timestamp: float = 0.0
    connected: bool = False

    def get_combined_altitude(self) -> float:
        """Get combined altitude command from triggers."""
        # Right trigger = ascend (positive), Left trigger = descend (negative)
        return self.right_trigger - self.left_trigger


@dataclass
class InputConfig:
    """Configuration for input handling."""

    # Dead zones (0-1)
    left_stick_deadzone: float = 0.15
    right_stick_deadzone: float = 0.15
    trigger_deadzone: float = 0.05

    # Sensitivity multipliers
    yaw_sensitivity: float = 1.0
    pitch_sensitivity: float = 1.0
    roll_sensitivity: float = 1.0
    throttle_sensitivity: float = 1.0
    altitude_sensitivity: float = 1.0

    # Keyboard specific
    keyboard_ramp_rate: float = 3.0  # How fast keyboard inputs ramp up/down

    # Input smoothing (exponential moving average)
    smoothing_factor: float = 0.3  # 0 = no smoothing, 1 = max smoothing

    # Response curves
    use_exponential_response: bool = True
    exponential_factor: float = 2.0  # Higher = more exponential


class XboxControllerHandler:
    """
    Handler for Xbox Wireless Controller input via XInput.

    The Xbox Wireless Controller is connected via:
    - USB cable
    - Xbox Wireless Adapter for Windows
    - Bluetooth (Windows 10/11)

    XInput provides low-latency access to controller state with:
    - Thumbstick positions (16-bit signed)
    - Trigger values (8-bit unsigned)
    - Digital button states
    - Vibration feedback
    """

    # XInput button mappings
    BUTTON_MAP = {
        "DPAD_UP": "dpad_up",
        "DPAD_DOWN": "dpad_down",
        "DPAD_LEFT": "dpad_left",
        "DPAD_RIGHT": "dpad_right",
        "START": "button_start",
        "BACK": "button_back",
        "LEFT_THUMB": "button_l3",
        "RIGHT_THUMB": "button_r3",
        "LEFT_SHOULDER": "button_lb",
        "RIGHT_SHOULDER": "button_rb",
        "A": "button_a",
        "B": "button_b",
        "X": "button_x",
        "Y": "button_y",
    }

    def __init__(self, controller_index: int = 0, config: Optional[InputConfig] = None):
        """
        Initialize Xbox controller handler.

        Args:
            controller_index: XInput controller index (0-3)
            config: Input configuration
        """
        self.controller_index = controller_index
        self.config = config if config is not None else InputConfig()
        self.connected = False
        self.last_state = ControllerState(source=InputSource.XBOX_CONTROLLER)

        # Smoothed values
        self._smooth_left_x = 0.0
        self._smooth_left_y = 0.0
        self._smooth_right_x = 0.0
        self._smooth_right_y = 0.0

        # Button edge detection
        self._prev_buttons: Dict[str, bool] = {}

    def is_available(self) -> bool:
        """Check if XInput is available."""
        return XINPUT_AVAILABLE

    def check_connection(self) -> bool:
        """Check if controller is connected."""
        if not XINPUT_AVAILABLE or XInput is None:
            return False
        try:
            assert XInput is not None
            state = XInput.get_state(self.controller_index)
            self.connected = True
            return True
        except XInput.XInputNotConnectedError:
            self.connected = False
            return False

    def _apply_deadzone(self, value: float, deadzone: float) -> float:
        """Apply deadzone and rescale the value."""
        if abs(value) < deadzone:
            return 0.0
        # Rescale so the output starts at 0 after the deadzone
        sign = 1 if value > 0 else -1
        return sign * (abs(value) - deadzone) / (1.0 - deadzone)

    def _apply_response_curve(self, value: float) -> float:
        """Apply exponential response curve for finer control at low values."""
        if not self.config.use_exponential_response:
            return value

        sign = 1 if value >= 0 else -1
        magnitude = abs(value)
        # Exponential curve: output = sign(x) * |x|^exponent
        curved = magnitude**self.config.exponential_factor
        return sign * curved

    def _smooth_value(self, current: float, previous: float) -> float:
        """Apply exponential moving average smoothing."""
        alpha = self.config.smoothing_factor
        return alpha * previous + (1 - alpha) * current

    def poll(self) -> ControllerState:
        """
        Poll the controller for current state.

        Returns:
            Normalized controller state
        """
        if not XINPUT_AVAILABLE or XInput is None:
            return ControllerState(source=InputSource.NONE, connected=False)

        try:
            # Get raw state from XInput
            assert XInput is not None
            state = XInput.get_state(self.controller_index)
            self.connected = True

            # Get thumbstick values (normalized to -1 to 1 by XInput)
            thumb_values = XInput.get_thumb_values(state)
            left_x_raw, left_y_raw = thumb_values[0]
            right_x_raw, right_y_raw = thumb_values[1]

            # Get trigger values (normalized to 0 to 1)
            trigger_values = XInput.get_trigger_values(state)
            left_trigger, right_trigger = trigger_values

            # Get button states
            buttons = XInput.get_button_values(state)

            # Apply deadzones
            left_x = self._apply_deadzone(left_x_raw, self.config.left_stick_deadzone)
            left_y = self._apply_deadzone(left_y_raw, self.config.left_stick_deadzone)
            right_x = self._apply_deadzone(
                right_x_raw, self.config.right_stick_deadzone
            )
            right_y = self._apply_deadzone(
                right_y_raw, self.config.right_stick_deadzone
            )
            lt = self._apply_deadzone(left_trigger, self.config.trigger_deadzone)
            rt = self._apply_deadzone(right_trigger, self.config.trigger_deadzone)

            # Apply response curves
            left_x = self._apply_response_curve(left_x)
            left_y = self._apply_response_curve(left_y)
            right_x = self._apply_response_curve(right_x)
            right_y = self._apply_response_curve(right_y)

            # Apply smoothing
            self._smooth_left_x = self._smooth_value(left_x, self._smooth_left_x)
            self._smooth_left_y = self._smooth_value(left_y, self._smooth_left_y)
            self._smooth_right_x = self._smooth_value(right_x, self._smooth_right_x)
            self._smooth_right_y = self._smooth_value(right_y, self._smooth_right_y)

            # Apply sensitivity
            yaw = self._smooth_left_x * self.config.yaw_sensitivity
            throttle = self._smooth_left_y * self.config.throttle_sensitivity
            roll = self._smooth_right_x * self.config.roll_sensitivity
            pitch = self._smooth_right_y * self.config.pitch_sensitivity
            lt_scaled = lt * self.config.altitude_sensitivity
            rt_scaled = rt * self.config.altitude_sensitivity

            # Create state object
            self.last_state = ControllerState(
                left_stick_x=yaw,
                left_stick_y=throttle,
                right_stick_x=roll,
                right_stick_y=pitch,
                left_trigger=lt_scaled,
                right_trigger=rt_scaled,
                button_a=buttons.get("A", False),
                button_b=buttons.get("B", False),
                button_x=buttons.get("X", False),
                button_y=buttons.get("Y", False),
                button_start=buttons.get("START", False),
                button_back=buttons.get("BACK", False),
                button_lb=buttons.get("LEFT_SHOULDER", False),
                button_rb=buttons.get("RIGHT_SHOULDER", False),
                dpad_up=buttons.get("DPAD_UP", False),
                dpad_down=buttons.get("DPAD_DOWN", False),
                dpad_left=buttons.get("DPAD_LEFT", False),
                dpad_right=buttons.get("DPAD_RIGHT", False),
                source=InputSource.XBOX_CONTROLLER,
                timestamp=time.perf_counter(),
                connected=True,
            )

            # Update previous button states for edge detection
            self._prev_buttons = buttons.copy()

            return self.last_state

        except XInput.XInputNotConnectedError:
            self.connected = False
            return ControllerState(source=InputSource.NONE, connected=False)

    def set_vibration(self, left_motor: float, right_motor: float):
        """
        Set controller vibration (haptic feedback).

        Args:
            left_motor: Left motor intensity (0-1)
            right_motor: Right motor intensity (0-1)
        """
        if XINPUT_AVAILABLE and self.connected and XInput is not None:
            try:
                XInput.set_vibration(
                    self.controller_index,
                    int(left_motor * 65535),
                    int(right_motor * 65535),
                )
            except:
                pass

    def stop_vibration(self):
        """Stop all vibration."""
        self.set_vibration(0, 0)


class KeyboardHandler:
    """
    Keyboard input handler as fallback when no controller is available.

    Key mappings:
    - W/S: Pitch forward/backward
    - A/D: Roll left/right (or Yaw with WASD mode)
    - Q/E: Yaw left/right
    - Space: Ascend
    - Left Shift: Descend
    - Arrow keys: Alternative movement
    - Enter: Arm/Disarm
    - Escape: Emergency stop
    - R: Reset simulation

    The keyboard handler simulates analog input by ramping values
    up and down smoothly, avoiding the "binary" feel of digital keys.
    """

    # Default key mappings
    KEY_MAP = {
        # Movement (WASD)
        "w": "pitch_forward",
        "s": "pitch_backward",
        "a": "yaw_left",
        "d": "yaw_right",
        "q": "roll_left",
        "e": "roll_right",
        # Altitude
        "space": "ascend",
        "shift": "descend",
        # Arrow keys (alternative)
        "up": "pitch_forward",
        "down": "pitch_backward",
        "left": "yaw_left",
        "right": "yaw_right",
        # Controls
        "enter": "arm_toggle",
        "esc": "emergency_stop",
        "r": "reset",
        "tab": "mode_toggle",
        # Sensitivity
        "[": "sensitivity_down",
        "]": "sensitivity_up",
    }

    def __init__(self, config: Optional[InputConfig] = None):
        """
        Initialize keyboard handler.

        Args:
            config: Input configuration
        """
        self.config = config if config is not None else InputConfig()
        self.last_state = ControllerState(source=InputSource.KEYBOARD)

        # Ramped values for smooth transitions
        self._ramp_pitch = 0.0
        self._ramp_roll = 0.0
        self._ramp_yaw = 0.0
        self._ramp_throttle = 0.0
        self._ramp_altitude = 0.0

        # Key states
        self._key_states: Dict[str, bool] = {}

        # Last update time for ramping
        self._last_update = time.perf_counter()

        # Button edge detection
        self._prev_arm = False
        self._prev_reset = False
        self._prev_mode = False

    def is_available(self) -> bool:
        """Check if keyboard library is available."""
        return KEYBOARD_AVAILABLE

    def _ramp_toward(self, current: float, target: float, dt: float) -> float:
        """Ramp a value toward target at configured rate."""
        rate = self.config.keyboard_ramp_rate
        max_change = rate * dt

        diff = target - current
        if abs(diff) <= max_change:
            return target

        return current + (max_change if diff > 0 else -max_change)

    def poll(self) -> ControllerState:
        """
        Poll keyboard for current state.

        Returns:
            Normalized controller state
        """
        if not KEYBOARD_AVAILABLE or keyboard is None:
            return ControllerState(source=InputSource.NONE, connected=False)

        current_time = time.perf_counter()
        dt = current_time - self._last_update
        self._last_update = current_time

        # Read key states
        assert keyboard is not None
        kb = keyboard
        assert kb is not None
        w_pressed = kb.is_pressed("w")
        s_pressed = kb.is_pressed("s")
        a_pressed = kb.is_pressed("a")
        d_pressed = kb.is_pressed("d")
        q_pressed = kb.is_pressed("q")
        e_pressed = kb.is_pressed("e")
        space_pressed = kb.is_pressed("space")
        shift_pressed = kb.is_pressed("shift")
        up_pressed = kb.is_pressed("up")
        down_pressed = kb.is_pressed("down")
        left_pressed = kb.is_pressed("left")
        right_pressed = kb.is_pressed("right")
        enter_pressed = kb.is_pressed("enter")
        esc_pressed = kb.is_pressed("esc")
        r_pressed = kb.is_pressed("r")
        tab_pressed = kb.is_pressed("tab")

        # Calculate target values from key states
        # Pitch: W/Up = forward (+1), S/Down = backward (-1)
        pitch_target = 0.0
        if w_pressed or up_pressed:
            pitch_target += 1.0
        if s_pressed or down_pressed:
            pitch_target -= 1.0

        # Yaw: A/Left = left (-1), D/Right = right (+1)
        yaw_target = 0.0
        if a_pressed or left_pressed:
            yaw_target -= 1.0
        if d_pressed or right_pressed:
            yaw_target += 1.0

        # Roll: Q = left (-1), E = right (+1)
        roll_target = 0.0
        if q_pressed:
            roll_target -= 1.0
        if e_pressed:
            roll_target += 1.0

        # Altitude: Space = ascend, Shift = descend
        altitude_target = 0.0
        if space_pressed:
            altitude_target += 1.0
        if shift_pressed:
            altitude_target -= 1.0

        # Ramp values for smooth response
        self._ramp_pitch = self._ramp_toward(self._ramp_pitch, pitch_target, dt)
        self._ramp_yaw = self._ramp_toward(self._ramp_yaw, yaw_target, dt)
        self._ramp_roll = self._ramp_toward(self._ramp_roll, roll_target, dt)
        self._ramp_altitude = self._ramp_toward(
            self._ramp_altitude, altitude_target, dt
        )

        # Apply sensitivity
        pitch = self._ramp_pitch * self.config.pitch_sensitivity
        yaw = self._ramp_yaw * self.config.yaw_sensitivity
        roll = self._ramp_roll * self.config.roll_sensitivity

        # Split altitude into trigger values
        rt = max(0, self._ramp_altitude) * self.config.altitude_sensitivity
        lt = max(0, -self._ramp_altitude) * self.config.altitude_sensitivity

        # Button edge detection (only trigger on press, not hold)
        arm_edge = enter_pressed and not self._prev_arm
        reset_edge = r_pressed and not self._prev_reset
        mode_edge = tab_pressed and not self._prev_mode

        self._prev_arm = enter_pressed
        self._prev_reset = r_pressed
        self._prev_mode = tab_pressed

        # Create state object
        # Note: Keyboard maps to right stick for pitch/roll since
        # left stick is typically throttle/yaw on a controller
        self.last_state = ControllerState(
            left_stick_x=yaw,  # Yaw control
            left_stick_y=0.0,  # Throttle (not used with keyboard)
            right_stick_x=roll,  # Roll control
            right_stick_y=pitch,  # Pitch control
            left_trigger=lt,  # Descend
            right_trigger=rt,  # Ascend
            button_a=arm_edge,  # Arm toggle (edge-triggered)
            button_b=esc_pressed,  # Emergency stop
            button_x=reset_edge,  # Reset (edge-triggered)
            button_y=mode_edge,  # Mode toggle (edge-triggered)
            button_start=False,
            button_back=False,
            button_lb=False,
            button_rb=False,
            dpad_up=False,
            dpad_down=False,
            dpad_left=False,
            dpad_right=False,
            source=InputSource.KEYBOARD,
            timestamp=current_time,
            connected=True,
        )

        return self.last_state


class UnifiedInputHandler:
    """
    Unified input handler that manages both Xbox controller and keyboard.

    This class provides a single interface for input regardless of the
    source, with automatic fallback from controller to keyboard if the
    controller is disconnected.

    Design Principles:
    1. Controller has priority when connected
    2. Seamless fallback to keyboard
    3. Consistent input scaling and response
    4. Real-time polling at configurable rate

    For real-time systems, the polling rate should be at least 2x
    the control loop rate to satisfy the Nyquist criterion for
    sampling. At 50Hz control, we poll at 100Hz minimum.
    """

    def __init__(
        self,
        config: Optional[InputConfig] = None,
        controller_index: int = 0,
        prefer_controller: bool = True,
    ):
        """
        Initialize unified input handler.

        Args:
            config: Input configuration
            controller_index: Xbox controller index (0-3)
            prefer_controller: If True, use controller when available
        """
        self.config = config if config is not None else InputConfig()
        self.prefer_controller = prefer_controller

        # Initialize handlers
        self.xbox = XboxControllerHandler(controller_index, self.config)
        self.keyboard = KeyboardHandler(self.config)

        # Current state
        self.current_state = ControllerState()
        self.active_source = InputSource.NONE

        # Connection check interval
        self._last_connection_check = 0.0
        self._connection_check_interval = 1.0  # Check every 1 second

        # Callbacks
        self._on_source_change: Optional[Callable[[InputSource], None]] = None
        self._on_button_press: Optional[Callable[[str], None]] = None

        # Statistics
        self.poll_count = 0
        self.total_poll_time = 0.0

    def set_on_source_change(self, callback: Callable[[InputSource], None]):
        """Set callback for when input source changes."""
        self._on_source_change = callback

    def set_on_button_press(self, callback: Callable[[str], None]):
        """Set callback for button press events."""
        self._on_button_press = callback

    def poll(self) -> ControllerState:
        """
        Poll for current input state.

        This method:
        1. Checks controller connection (periodically)
        2. Polls the active input source
        3. Falls back to keyboard if controller disconnected

        Returns:
            Current input state from active source
        """
        start_time = time.perf_counter()

        # Periodic connection check
        if start_time - self._last_connection_check > self._connection_check_interval:
            self._last_connection_check = start_time
            self._check_connections()

        # Poll based on active source preference
        old_source = self.active_source

        if self.prefer_controller and self.xbox.connected:
            self.current_state = self.xbox.poll()
            self.active_source = InputSource.XBOX_CONTROLLER
        elif self.keyboard.is_available():
            self.current_state = self.keyboard.poll()
            self.active_source = InputSource.KEYBOARD
        else:
            self.current_state = ControllerState()
            self.active_source = InputSource.NONE

        # Notify on source change
        if old_source != self.active_source and self._on_source_change:
            self._on_source_change(self.active_source)

        # Update statistics
        self.poll_count += 1
        self.total_poll_time += time.perf_counter() - start_time

        return self.current_state

    def _check_connections(self):
        """Check connection status of all input sources."""
        if self.xbox.is_available():
            self.xbox.check_connection()

    def get_flight_commands(self) -> Dict[str, float]:
        """
        Convert current input state to flight commands.

        Returns:
            Dictionary with normalized flight commands:
            - pitch: -1 to +1 (back to forward)
            - roll: -1 to +1 (left to right)
            - yaw_rate: -1 to +1 (left to right)
            - throttle: -1 to +1 (down to up)
            - altitude_rate: -1 to +1 (descend to ascend)
        """
        state = self.current_state

        return {
            "pitch": state.right_stick_y,
            "roll": state.right_stick_x,
            "yaw_rate": state.left_stick_x,
            "throttle": state.left_stick_y,
            "altitude_rate": state.get_combined_altitude(),
        }

    def get_button_events(self) -> Dict[str, bool]:
        """
        Get current button states for event handling.

        Returns:
            Dictionary of button states
        """
        state = self.current_state
        return {
            "arm_toggle": state.button_a,
            "emergency_stop": state.button_b,
            "reset": state.button_x,
            "mode_toggle": state.button_y,
            "start": state.button_start,
            "back": state.button_back,
            "sensitivity_down": state.button_lb,
            "sensitivity_up": state.button_rb,
        }

    def set_vibration(self, left: float, right: float):
        """Set controller vibration if available."""
        if self.active_source == InputSource.XBOX_CONTROLLER:
            self.xbox.set_vibration(left, right)

    def stop_vibration(self):
        """Stop controller vibration."""
        self.xbox.stop_vibration()

    def get_average_poll_time(self) -> float:
        """Get average polling time in milliseconds."""
        if self.poll_count == 0:
            return 0.0
        return (self.total_poll_time / self.poll_count) * 1000

    def get_status(self) -> Dict:
        """Get current status for display."""
        return {
            "active_source": self.active_source.name,
            "controller_connected": self.xbox.connected,
            "keyboard_available": self.keyboard.is_available(),
            "poll_count": self.poll_count,
            "avg_poll_time_ms": self.get_average_poll_time(),
        }

    def update_config(self, **kwargs):
        """
        Update input configuration at runtime.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Propagate to handlers
        self.xbox.config = self.config
        self.keyboard.config = self.config


# Utility function for testing
def print_input_state(state: ControllerState):
    """Print controller state for debugging."""
    print(f"\n{'='*50}")
    print(f"Input Source: {state.source.name}")
    print(f"Connected: {state.connected}")
    print(f"Timestamp: {state.timestamp:.3f}")
    print(f"\nSticks:")
    print(f"  Left:  X={state.left_stick_x:+.3f}  Y={state.left_stick_y:+.3f}")
    print(f"  Right: X={state.right_stick_x:+.3f}  Y={state.right_stick_y:+.3f}")
    print(f"\nTriggers:")
    print(f"  LT={state.left_trigger:.3f}  RT={state.right_trigger:.3f}")
    print(f"  Combined Altitude: {state.get_combined_altitude():+.3f}")
    print(f"\nButtons:")
    print(
        f"  A={state.button_a}  B={state.button_b}  X={state.button_x}  Y={state.button_y}"
    )


if __name__ == "__main__":
    print("Input Handler Test")
    print("=" * 50)
    print(f"XInput Available: {XINPUT_AVAILABLE}")
    print(f"Keyboard Available: {KEYBOARD_AVAILABLE}")

    # Create unified handler
    handler = UnifiedInputHandler()

    print("\nPolling input for 10 seconds...")
    print("Use controller or keyboard to test input")
    print("Controller: Sticks, Triggers, Buttons")
    print("Keyboard: WASD, Space/Shift, Enter, Esc")

    start_time = time.perf_counter()
    last_print = start_time

    try:
        while time.perf_counter() - start_time < 10.0:
            # Poll at 100Hz
            state = handler.poll()

            # Print every 0.5 seconds
            if time.perf_counter() - last_print > 0.5:
                last_print = time.perf_counter()
                print_input_state(state)

                # Print flight commands
                commands = handler.get_flight_commands()
                print("\nFlight Commands:")
                for name, value in commands.items():
                    print(f"  {name}: {value:+.3f}")

            time.sleep(0.01)  # 100Hz polling

    except KeyboardInterrupt:
        print("\nTest interrupted")

    # Print statistics
    print("\n" + "=" * 50)
    print("Statistics:")
    status = handler.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    print("\nInput handler test complete!")
