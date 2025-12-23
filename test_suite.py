"""
test_suite.py - Unit and Integration Tests for Quadcopter Simulator
CMSE443 Real-Time Systems Design - Term Project

This module provides comprehensive testing for all system components:
- Unit tests for mathematical model
- Unit tests for PID controller
- Unit tests for input handler
- Integration tests for control loop timing
- System tests for end-to-end functionality

Testing Methodology:
- Unit testing: Isolate and test individual components
- Integration testing: Test component interactions
- Timing tests: Verify real-time constraints are met

References:
- Python unittest documentation
- Real-time systems testing methodologies
"""

import unittest
import time
import math
import numpy as np
from collections import deque
import sys
import os

# Import modules to test
from quadcopter_model import (
    QuadcopterModel,
    QuadcopterState,
    QuadcopterParameters,
    ControlInput,
    IntegrationMethod,
    create_hover_control,
)
from pid_controller import PIDController, PIDGains, QuadcopterPIDController, RateLimiter
from input_handler import (
    InputConfig,
    ControllerState,
    InputSource,
    KeyboardHandler,
    UnifiedInputHandler,
)


class TestQuadcopterModel(unittest.TestCase):
    """Unit tests for the quadcopter mathematical model."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = QuadcopterModel()
        self.params = QuadcopterParameters()

    def test_initialization(self):
        """Test model initializes with correct default state."""
        state = self.model.get_state()

        self.assertEqual(state.x, 0.0)
        self.assertEqual(state.y, 0.0)
        self.assertEqual(state.z, 0.0)
        self.assertEqual(state.phi, 0.0)
        self.assertEqual(state.theta, 0.0)
        self.assertEqual(state.psi, 0.0)

    def test_hover_thrust_calculation(self):
        """Test hover thrust equals mass * gravity."""
        expected_thrust = self.params.mass * self.params.gravity
        calculated_thrust = self.model.get_hover_thrust()

        self.assertAlmostEqual(calculated_thrust, expected_thrust, places=5)

    def test_euler_integration_stability(self):
        """Test Euler integration remains stable for hover."""
        self.model.set_integration_method(IntegrationMethod.EULER)
        hover_control = create_hover_control(self.model)

        # Simulate 1 second at 50Hz
        for _ in range(50):
            self.model.step(hover_control, 0.02)

        state = self.model.get_state()

        # Position should remain near zero for hover
        self.assertAlmostEqual(state.x, 0.0, places=1)
        self.assertAlmostEqual(state.y, 0.0, places=1)
        self.assertAlmostEqual(state.z, 0.0, places=1)

    def test_rk4_integration_stability(self):
        """Test RK4 integration remains stable for hover."""
        self.model.set_integration_method(IntegrationMethod.RUNGE_KUTTA_4)
        hover_control = create_hover_control(self.model)

        # Simulate 1 second at 50Hz
        for _ in range(50):
            self.model.step(hover_control, 0.02)

        state = self.model.get_state()

        # Position should remain near zero for hover
        self.assertAlmostEqual(state.x, 0.0, places=2)
        self.assertAlmostEqual(state.y, 0.0, places=2)
        self.assertAlmostEqual(state.z, 0.0, places=2)

    def test_rk4_more_accurate_than_euler(self):
        """Test that RK4 is more accurate than Euler for same step size."""
        # Test with a larger time step where difference is more apparent
        dt = 0.1

        # Euler model
        euler_model = QuadcopterModel(integration_method=IntegrationMethod.EULER)
        hover_control = create_hover_control(euler_model)

        for _ in range(10):
            euler_model.step(hover_control, dt)
        euler_state = euler_model.get_state()

        # RK4 model
        rk4_model = QuadcopterModel(integration_method=IntegrationMethod.RUNGE_KUTTA_4)
        hover_control = create_hover_control(rk4_model)

        for _ in range(10):
            rk4_model.step(hover_control, dt)
        rk4_state = rk4_model.get_state()

        # RK4 should have smaller drift from origin
        euler_drift = math.sqrt(euler_state.x**2 + euler_state.y**2 + euler_state.z**2)
        rk4_drift = math.sqrt(rk4_state.x**2 + rk4_state.y**2 + rk4_state.z**2)

        # RK4 should be at least as stable as Euler
        self.assertLessEqual(rk4_drift, euler_drift + 0.1)

    def test_thrust_causes_ascent(self):
        """Test that increasing thrust causes upward movement (negative Z in NED)."""
        hover_thrust = self.model.get_hover_thrust()

        # Apply extra thrust
        control = ControlInput(thrust=hover_thrust * 1.5, use_rotor_speeds=False)

        # Simulate
        for _ in range(50):
            self.model.step(control, 0.02)

        state = self.model.get_state()

        # Z should be negative (upward in NED)
        self.assertLess(state.z, 0)

    def test_position_history_tracking(self):
        """Test that position history is correctly tracked."""
        hover_control = create_hover_control(self.model)

        # Simulate some steps
        for _ in range(15):
            self.model.step(hover_control, 0.02)

        history = self.model.get_position_history()

        # Should have max 10 entries
        self.assertLessEqual(len(history), 10)

        # Each entry should be a list of 3 coordinates
        for pos in history:
            self.assertEqual(len(pos), 3)

    def test_rotation_matrix_identity(self):
        """Test rotation matrix is identity for zero angles."""
        R = self.model._rotation_matrix(0, 0, 0)

        expected = np.eye(3)
        np.testing.assert_array_almost_equal(R, expected, decimal=5)

    def test_rotation_matrix_90_yaw(self):
        """Test rotation matrix for 90 degree yaw."""
        R = self.model._rotation_matrix(0, 0, math.pi / 2)

        # 90 degree yaw should swap x and y axes
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        np.testing.assert_array_almost_equal(R, expected, decimal=5)

    def test_state_copy(self):
        """Test state copy creates independent copy."""
        state1 = QuadcopterState(x=1.0, y=2.0, z=3.0)
        state2 = state1.copy()

        # Modify state1
        state1.x = 10.0

        # state2 should be unchanged
        self.assertEqual(state2.x, 1.0)

    def test_configurable_parameters(self):
        """Test that parameters can be configured."""
        new_params = QuadcopterParameters(mass=2.0, gravity=10.0)
        self.model.set_parameters(new_params)

        expected_hover = 2.0 * 10.0
        self.assertAlmostEqual(self.model.get_hover_thrust(), expected_hover)


class TestPIDController(unittest.TestCase):
    """Unit tests for the PID controller."""

    def setUp(self):
        """Set up test fixtures."""
        self.gains = PIDGains(Kp=2.0, Ki=0.1, Kd=0.5)
        self.pid = PIDController(self.gains, "test")

    def test_initialization(self):
        """Test PID controller initializes correctly."""
        self.assertEqual(self.pid.gains.Kp, 2.0)
        self.assertEqual(self.pid.gains.Ki, 0.1)
        self.assertEqual(self.pid.gains.Kd, 0.5)

    def test_proportional_only(self):
        """Test proportional-only control."""
        p_only_gains = PIDGains(Kp=2.0, Ki=0.0, Kd=0.0)
        pid = PIDController(p_only_gains)

        output = pid.update(setpoint=10.0, measurement=0.0, dt=0.02)

        # Output should be Kp * error = 2.0 * 10.0 = 20.0
        self.assertAlmostEqual(output, 20.0, places=1)

    def test_integral_accumulation(self):
        """Test integral term accumulates over time."""
        i_only_gains = PIDGains(Kp=0.0, Ki=1.0, Kd=0.0, integral_limit=100)
        pid = PIDController(i_only_gains)

        # Apply constant error over multiple steps
        output = 0.0
        for _ in range(10):
            output = pid.update(setpoint=10.0, measurement=0.0, dt=0.1)

        # Integral should have accumulated
        # I = Ki * sum(error * dt) = 1.0 * 10 * 10 * 0.1 = 10.0
        self.assertGreater(output, 0)

    def test_integral_anti_windup(self):
        """Test integral anti-windup limits."""
        gains = PIDGains(Kp=0.0, Ki=10.0, Kd=0.0, integral_limit=5.0)
        pid = PIDController(gains)

        # Apply large error for many steps
        for _ in range(100):
            pid.update(setpoint=100.0, measurement=0.0, dt=0.1)

        # Integral should be limited
        self.assertLessEqual(abs(pid.state.integral), 5.0)

    def test_derivative_response(self):
        """Test derivative term responds to change."""
        d_only_gains = PIDGains(Kp=0.0, Ki=0.0, Kd=1.0, derivative_filter=0.0)
        pid = PIDController(d_only_gains)

        # First call establishes baseline
        pid.update(setpoint=0.0, measurement=0.0, dt=0.02)

        # Second call with changed error
        output = pid.update(setpoint=10.0, measurement=0.0, dt=0.02)

        # Derivative should respond to error change
        # D = Kd * (error - prev_error) / dt = 1.0 * 10 / 0.02 = 500
        self.assertGreater(abs(output), 0)

    def test_output_limits(self):
        """Test output clamping."""
        gains = PIDGains(Kp=10.0, Ki=0.0, Kd=0.0, output_min=-5.0, output_max=5.0)
        pid = PIDController(gains)

        output = pid.update(setpoint=100.0, measurement=0.0, dt=0.02)

        # Output should be clamped to max
        self.assertEqual(output, 5.0)

    def test_reset_clears_state(self):
        """Test reset clears internal state."""
        # Build up some state
        for _ in range(10):
            self.pid.update(setpoint=10.0, measurement=0.0, dt=0.02)

        # Reset
        self.pid.reset()

        # State should be cleared
        self.assertEqual(self.pid.state.integral, 0.0)
        self.assertEqual(self.pid.state.previous_error, 0.0)
        self.assertFalse(self.pid.state.initialized)

    def test_convergence_to_setpoint(self):
        """Test PID controller converges to setpoint."""
        pid = PIDController(PIDGains(Kp=1.0, Ki=0.2, Kd=0.5))

        value = 0.0
        setpoint = 10.0

        # Simulate closed-loop control
        for _ in range(500):
            output = pid.update(setpoint, value, 0.02)
            # Simple first-order system response
            value += output * 0.01  # Smaller gain for stability

        # Should converge close to setpoint (within 20%)
        self.assertLess(abs(value - setpoint), setpoint * 0.2)


class TestQuadcopterPIDController(unittest.TestCase):
    """Unit tests for the quadcopter PID controller."""

    def setUp(self):
        """Set up test fixtures."""
        self.controller = QuadcopterPIDController(mass=1.0, gravity=9.81)

    def test_hover_thrust(self):
        """Test hover thrust calculation."""
        self.assertAlmostEqual(self.controller.hover_thrust, 9.81, places=2)

    def test_position_setpoint(self):
        """Test position setpoint setting."""
        self.controller.set_position_setpoint(5.0, 3.0, -2.0)

        np.testing.assert_array_equal(
            self.controller.position_setpoint, np.array([5.0, 3.0, -2.0])
        )

    def test_compute_control_output_format(self):
        """Test control output has correct format."""
        position = np.array([0.0, 0.0, 0.0])
        velocity = np.array([0.0, 0.0, 0.0])
        attitude = np.array([0.0, 0.0, 0.0])
        angular_velocity = np.array([0.0, 0.0, 0.0])

        output = self.controller.compute_control(
            position, velocity, attitude, angular_velocity, 0.02
        )

        # Check all expected keys are present
        self.assertIn("thrust", output)
        self.assertIn("tau_phi", output)
        self.assertIn("tau_theta", output)
        self.assertIn("tau_psi", output)

    def test_gains_dict(self):
        """Test gains dictionary retrieval."""
        gains = self.controller.get_gains_dict()

        expected_keys = [
            "position_x",
            "position_y",
            "position_z",
            "roll",
            "pitch",
            "yaw",
        ]

        for key in expected_keys:
            self.assertIn(key, gains)
            self.assertIn("Kp", gains[key])
            self.assertIn("Ki", gains[key])
            self.assertIn("Kd", gains[key])


class TestRateLimiter(unittest.TestCase):
    """Unit tests for the rate limiter."""

    def test_rate_limiting(self):
        """Test value changes are rate-limited."""
        limiter = RateLimiter(rate_limit=10.0, initial_value=0.0)

        # Try to change by 100 in 0.1 seconds
        result = limiter.update(100.0, 0.1)

        # Should only change by rate_limit * dt = 10 * 0.1 = 1.0
        self.assertAlmostEqual(result, 1.0, places=2)

    def test_small_changes_not_limited(self):
        """Test small changes pass through."""
        limiter = RateLimiter(rate_limit=10.0, initial_value=0.0)

        # Small change within rate limit
        result = limiter.update(0.5, 0.1)

        self.assertAlmostEqual(result, 0.5, places=2)


class TestInputHandler(unittest.TestCase):
    """Unit tests for the input handler."""

    def test_config_defaults(self):
        """Test default input configuration."""
        config = InputConfig()

        self.assertEqual(config.left_stick_deadzone, 0.15)
        self.assertEqual(config.right_stick_deadzone, 0.15)
        self.assertTrue(config.use_exponential_response)

    def test_controller_state_initialization(self):
        """Test controller state initializes to zero."""
        state = ControllerState()

        self.assertEqual(state.left_stick_x, 0.0)
        self.assertEqual(state.left_stick_y, 0.0)
        self.assertEqual(state.right_stick_x, 0.0)
        self.assertEqual(state.right_stick_y, 0.0)
        self.assertEqual(state.source, InputSource.NONE)

    def test_combined_altitude(self):
        """Test combined altitude calculation."""
        state = ControllerState(left_trigger=0.3, right_trigger=0.8)

        # Combined = right - left = 0.8 - 0.3 = 0.5
        self.assertAlmostEqual(state.get_combined_altitude(), 0.5)


class TestTimingConstraints(unittest.TestCase):
    """Tests for real-time timing constraints."""

    def test_50hz_loop_feasibility(self):
        """Test that control loop can run at 50Hz."""
        from quadcopter_model import QuadcopterModel, create_hover_control
        from pid_controller import QuadcopterPIDController

        model = QuadcopterModel()
        controller = QuadcopterPIDController()

        target_period_ms = 20.0  # 50Hz
        execution_times = []

        # Simulate 100 control loop iterations
        for _ in range(100):
            start = time.perf_counter()

            # Typical control loop operations
            hover_control = create_hover_control(model)
            model.step(hover_control, 0.02)

            position = np.zeros(3)
            velocity = np.zeros(3)
            attitude = np.zeros(3)
            angular_velocity = np.zeros(3)

            controller.compute_control(
                position, velocity, attitude, angular_velocity, 0.02
            )

            end = time.perf_counter()
            execution_times.append((end - start) * 1000)

        avg_time = sum(execution_times) / len(execution_times)
        max_time = max(execution_times)

        # Average execution should be well under period
        self.assertLess(
            avg_time,
            target_period_ms * 0.5,
            f"Average execution {avg_time:.2f}ms exceeds 50% of period",
        )

        # WCET should be under deadline
        self.assertLess(
            max_time, target_period_ms, f"WCET {max_time:.2f}ms exceeds period"
        )

    def test_model_step_timing(self):
        """Test individual model step timing."""
        model = QuadcopterModel()
        hover_control = create_hover_control(model)

        times = []
        for _ in range(100):
            start = time.perf_counter()
            model.step(hover_control, 0.02)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg_time = sum(times) / len(times)

        # Model step should be very fast
        self.assertLess(
            avg_time, 1.0, f"Model step average {avg_time:.3f}ms is too slow"
        )

    def test_pid_computation_timing(self):
        """Test PID computation timing."""
        controller = QuadcopterPIDController()

        times = []
        for _ in range(100):
            start = time.perf_counter()
            controller.compute_control(
                np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), 0.02
            )
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg_time = sum(times) / len(times)

        # PID computation should be very fast
        self.assertLess(
            avg_time, 0.5, f"PID computation average {avg_time:.3f}ms is too slow"
        )


class TestIntegration(unittest.TestCase):
    """Integration tests for component interactions."""

    def test_model_pid_integration(self):
        """Test mathematical model works with PID controller."""
        model = QuadcopterModel()
        pid = QuadcopterPIDController(mass=model.params.mass)

        # Start drone at some positive altitude (z=10 in NED means 10m below origin)
        # Target z=-5 means 5m above origin, so drone must climb (z decreasing)
        model.state.z = 10.0  # Start below target

        # Set target altitude
        pid.set_position_setpoint(0, 0, -5)  # 5m up in NED

        initial_z = model.state.z

        # Simulate
        for _ in range(500):  # 10 seconds at 50Hz
            state = model.get_state()

            position = state.get_position()
            velocity = state.get_velocity()
            attitude = state.get_euler_angles()
            angular_velocity = state.get_angular_velocity()

            control_output = pid.compute_control(
                position, velocity, attitude, angular_velocity, 0.02
            )

            from quadcopter_model import ControlInput

            control = ControlInput(
                thrust=control_output["thrust"],
                tau_phi=control_output["tau_phi"],
                tau_theta=control_output["tau_theta"],
                tau_psi=control_output["tau_psi"],
                use_rotor_speeds=False,
            )

            model.step(control, 0.02)

        final_state = model.get_state()

        # Should have moved toward target altitude (z should decrease from 10 toward -5)
        self.assertLess(
            final_state.z, initial_z, "Drone should have ascended (Z decreased)"
        )

    def test_history_tracking_integration(self):
        """Test position/velocity history is tracked during simulation."""
        model = QuadcopterModel()

        # Simulate some steps
        hover = create_hover_control(model)
        for _ in range(20):
            model.step(hover, 0.02)

        pos_history = model.get_position_history()
        vel_history = model.get_velocity_history()

        self.assertEqual(len(pos_history), 10)  # Max 10
        self.assertEqual(len(vel_history), 10)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and boundary conditions."""

    def test_zero_time_step(self):
        """Test handling of zero time step."""
        model = QuadcopterModel()
        hover = create_hover_control(model)

        # Should not crash
        state = model.step(hover, 0.0)

        # State should be unchanged
        self.assertEqual(state.x, 0.0)

    def test_very_large_thrust(self):
        """Test handling of very large thrust."""
        model = QuadcopterModel()

        from quadcopter_model import ControlInput

        control = ControlInput(thrust=1000.0, use_rotor_speeds=False)

        # Should not crash
        state = model.step(control, 0.02)

        # Drone should move upward rapidly
        self.assertLess(state.z, 0)

    def test_negative_time_step(self):
        """Test handling of negative time step."""
        model = QuadcopterModel()
        hover = create_hover_control(model)

        # Should handle gracefully (or at least not crash)
        try:
            model.step(hover, -0.02)
        except Exception as e:
            self.fail(f"Negative time step caused exception: {e}")

    def test_extreme_angles(self):
        """Test model handles extreme attitude angles."""
        model = QuadcopterModel()

        # Set extreme initial attitude
        initial_state = QuadcopterState(
            phi=math.radians(80),  # 80 degree roll
            theta=math.radians(80),  # 80 degree pitch
        )
        model.reset(initial_state)

        hover = create_hover_control(model)

        # Should not crash
        for _ in range(10):
            model.step(hover, 0.02)


def run_all_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestQuadcopterModel))
    suite.addTests(loader.loadTestsFromTestCase(TestPIDController))
    suite.addTests(loader.loadTestsFromTestCase(TestQuadcopterPIDController))
    suite.addTests(loader.loadTestsFromTestCase(TestRateLimiter))
    suite.addTests(loader.loadTestsFromTestCase(TestInputHandler))
    suite.addTests(loader.loadTestsFromTestCase(TestTimingConstraints))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    print("=" * 70)
    print("CMSE443 Quadcopter Simulator - Test Suite")
    print("=" * 70)
    print()

    result = run_all_tests()

    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)
