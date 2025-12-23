"""
quadcopter_model.py - Mathematical Model of Quadcopter Dynamics
CMSE443 Real-Time Systems Design - Term Project

This module implements the mathematical model of a quadcopter using:
- Newton-Euler equations of motion
- Euler and Runge-Kutta (RK4) numerical integration methods
- Configurable physical parameters

Reference:
- Luukkonen, T. (2011). Modelling and control of quadcopter. Aalto University.
- Beard, R. W., & McLain, T. W. (2012). Small unmanned aircraft: Theory and practice.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Callable, Union
from enum import Enum
import time


class IntegrationMethod(Enum):
    """Numerical integration methods available for simulation."""

    EULER = "euler"
    RUNGE_KUTTA_4 = "rk4"


@dataclass
class QuadcopterParameters:
    """
    Physical parameters of the quadcopter.
    All units are SI (kg, m, s, rad).

    These parameters can be configured at runtime to test different
    quadcopter configurations and observe their effects on stability.
    """

    # Mass properties
    mass: float = 1.0  # Total mass [kg]

    # Inertia tensor (diagonal elements for symmetric quadcopter)
    Ixx: float = 0.0086  # Moment of inertia about x-axis [kg·m²]
    Iyy: float = 0.0086  # Moment of inertia about y-axis [kg·m²]
    Izz: float = 0.0172  # Moment of inertia about z-axis [kg·m²]

    # Geometry
    arm_length: float = 0.225  # Distance from center to rotor [m]

    # Aerodynamic coefficients
    thrust_coeff: float = 2.98e-6  # Thrust coefficient k_t [N/(rad/s)²]
    drag_coeff: float = 1.14e-7  # Drag coefficient k_d [N·m/(rad/s)²]

    # Environmental
    gravity: float = 9.81  # Gravitational acceleration [m/s²]
    air_density: float = 1.225  # Air density at sea level [kg/m³]

    # Drag coefficients for translational motion
    drag_coeff_x: float = 0.25  # Translational drag coefficient x
    drag_coeff_y: float = 0.25  # Translational drag coefficient y
    drag_coeff_z: float = 0.25  # Translational drag coefficient z

    # Motor properties
    max_rotor_speed: float = 1000.0  # Maximum rotor angular velocity [rad/s]
    min_rotor_speed: float = 0.0  # Minimum rotor angular velocity [rad/s]
    motor_time_constant: float = 0.02  # Motor response time constant [s]

    def get_inertia_matrix(self) -> np.ndarray:
        """Return the 3x3 inertia tensor matrix."""
        return np.diag([self.Ixx, self.Iyy, self.Izz])

    def get_inverse_inertia(self) -> np.ndarray:
        """Return the inverse of the inertia tensor."""
        return np.diag([1.0 / self.Ixx, 1.0 / self.Iyy, 1.0 / self.Izz])


@dataclass
class QuadcopterState:
    """
    Complete state vector of the quadcopter.

    The state is represented in the NED (North-East-Down) coordinate frame
    where +X points North, +Y points East, and +Z points Down.

    Euler angles follow the ZYX convention (yaw-pitch-roll).
    """

    # Position in world frame (NED) [m]
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # Velocity in world frame [m/s]
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    # Euler angles (ZYX convention) [rad]
    phi: float = 0.0  # Roll angle (rotation about x-axis)
    theta: float = 0.0  # Pitch angle (rotation about y-axis)
    psi: float = 0.0  # Yaw angle (rotation about z-axis)

    # Angular velocities in body frame [rad/s]
    p: float = 0.0  # Roll rate
    q: float = 0.0  # Pitch rate
    r: float = 0.0  # Yaw rate

    # Rotor speeds [rad/s] - for 4 rotors
    omega: np.ndarray = field(default_factory=lambda: np.zeros(4))

    # Timestamp
    timestamp: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert state to numpy array for numerical integration."""
        return np.array(
            [
                self.x,
                self.y,
                self.z,
                self.vx,
                self.vy,
                self.vz,
                self.phi,
                self.theta,
                self.psi,
                self.p,
                self.q,
                self.r,
            ]
        )

    @classmethod
    def from_array(
        cls, arr: np.ndarray, omega: Optional[np.ndarray] = None, timestamp: float = 0.0
    ) -> "QuadcopterState":
        """Create state from numpy array."""
        return cls(
            x=arr[0],
            y=arr[1],
            z=arr[2],
            vx=arr[3],
            vy=arr[4],
            vz=arr[5],
            phi=arr[6],
            theta=arr[7],
            psi=arr[8],
            p=arr[9],
            q=arr[10],
            r=arr[11],
            omega=omega if omega is not None else np.zeros(4),
            timestamp=timestamp,
        )

    def copy(self) -> "QuadcopterState":
        """Create a deep copy of the state."""
        return QuadcopterState(
            x=self.x,
            y=self.y,
            z=self.z,
            vx=self.vx,
            vy=self.vy,
            vz=self.vz,
            phi=self.phi,
            theta=self.theta,
            psi=self.psi,
            p=self.p,
            q=self.q,
            r=self.r,
            omega=self.omega.copy(),
            timestamp=self.timestamp,
        )

    def get_position(self) -> np.ndarray:
        """Return position vector [x, y, z]."""
        return np.array([self.x, self.y, self.z])

    def get_velocity(self) -> np.ndarray:
        """Return velocity vector [vx, vy, vz]."""
        return np.array([self.vx, self.vy, self.vz])

    def get_euler_angles(self) -> np.ndarray:
        """Return Euler angles [phi, theta, psi] in radians."""
        return np.array([self.phi, self.theta, self.psi])

    def get_angular_velocity(self) -> np.ndarray:
        """Return angular velocity vector [p, q, r] in body frame."""
        return np.array([self.p, self.q, self.r])


@dataclass
class ControlInput:
    """
    Control inputs to the quadcopter.

    These can be either direct rotor speed commands or
    higher-level thrust/moment commands.
    """

    # Total thrust [N] - sum of all rotor thrusts
    thrust: float = 0.0

    # Moments about body axes [N·m]
    tau_phi: float = 0.0  # Roll moment
    tau_theta: float = 0.0  # Pitch moment
    tau_psi: float = 0.0  # Yaw moment

    # Alternative: direct rotor speed commands [rad/s]
    omega_cmd: np.ndarray = field(default_factory=lambda: np.zeros(4))

    # Flag to indicate which input mode is used
    use_rotor_speeds: bool = False


class QuadcopterModel:
    """
    Mathematical model of quadcopter dynamics.

    This class implements the Newton-Euler equations of motion for a
    quadcopter and provides numerical integration using either Euler
    or 4th-order Runge-Kutta methods.

    The model follows the standard '+' configuration where:
    - Rotor 1 (front) and Rotor 3 (rear) spin clockwise
    - Rotor 2 (right) and Rotor 4 (left) spin counter-clockwise

    Coordinate System (NED - North-East-Down):
    - X-axis: Points North (forward in body frame)
    - Y-axis: Points East (right in body frame)
    - Z-axis: Points Down (positive Z is below the drone)

    Mathematical Model Reference:
    Luukkonen, T. (2011). "Modelling and control of quadcopter"
    """

    def __init__(
        self,
        params: Optional[QuadcopterParameters] = None,
        integration_method: IntegrationMethod = IntegrationMethod.RUNGE_KUTTA_4,
    ):
        """
        Initialize the quadcopter model.

        Args:
            params: Physical parameters (uses defaults if None)
            integration_method: Numerical integration method to use
        """
        self.params = params if params is not None else QuadcopterParameters()
        self.integration_method = integration_method
        self.state = QuadcopterState()

        # State history for visualization (last 10 values)
        self.position_history: list = []
        self.velocity_history: list = []
        self.max_history_length = 10

        # Simulation statistics
        self.total_steps = 0
        self.total_simulation_time = 0.0

    def reset(self, initial_state: Optional[QuadcopterState] = None):
        """Reset the model to initial state."""
        if initial_state is not None:
            self.state = initial_state.copy()
        else:
            self.state = QuadcopterState()

        self.position_history.clear()
        self.velocity_history.clear()
        self.total_steps = 0
        self.total_simulation_time = 0.0

    def _rotation_matrix(self, phi: float, theta: float, psi: float) -> np.ndarray:
        """
        Compute the rotation matrix from body frame to world frame.

        Uses ZYX Euler angle convention (yaw-pitch-roll).

        R = Rz(psi) * Ry(theta) * Rx(phi)

        Args:
            phi: Roll angle [rad]
            theta: Pitch angle [rad]
            psi: Yaw angle [rad]

        Returns:
            3x3 rotation matrix
        """
        cp, sp = np.cos(phi), np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        cy, sy = np.cos(psi), np.sin(psi)

        R = np.array(
            [
                [ct * cy, sp * st * cy - cp * sy, cp * st * cy + sp * sy],
                [ct * sy, sp * st * sy + cp * cy, cp * st * sy - sp * cy],
                [-st, sp * ct, cp * ct],
            ]
        )

        return R

    def _euler_rate_matrix(self, phi: float, theta: float) -> np.ndarray:
        """
        Compute the transformation matrix from body angular rates to Euler rates.

        [phi_dot  ]       [p]
        [theta_dot] = T * [q]
        [psi_dot  ]       [r]

        Note: This matrix is singular when theta = ±90°, which is a
        gimbal lock condition. The model assumes small angle deviations.

        Args:
            phi: Roll angle [rad]
            theta: Pitch angle [rad]

        Returns:
            3x3 transformation matrix
        """
        cp, sp = np.cos(phi), np.sin(phi)
        ct, tt = np.cos(theta), np.tan(theta)

        # Avoid division by zero near gimbal lock
        if abs(ct) < 1e-6:
            ct = 1e-6 if ct >= 0 else -1e-6

        T = np.array([[1, sp * tt, cp * tt], [0, cp, -sp], [0, sp / ct, cp / ct]])

        return T

    def _compute_thrust_and_moments(
        self, control: ControlInput
    ) -> Tuple[Union[float, np.floating], np.ndarray]:
        """
        Compute total thrust and moments from control input.

        For the '+' configuration quadcopter:
        - Rotor 1 (front, +X): CW, produces +roll moment
        - Rotor 2 (right, +Y): CCW, produces +pitch moment
        - Rotor 3 (rear, -X): CW, produces -roll moment
        - Rotor 4 (left, -Y): CCW, produces -pitch moment

        Thrust equation: F_i = k_t * omega_i^2
        Drag moment: M_i = k_d * omega_i^2

        Args:
            control: Control input (either thrust/moments or rotor speeds)

        Returns:
            Tuple of (total_thrust, moments_vector)
        """
        if control.use_rotor_speeds:
            # Compute from rotor speeds
            omega = control.omega_cmd
            k_t = self.params.thrust_coeff
            k_d = self.params.drag_coeff
            L = self.params.arm_length

            # Thrust from each rotor
            F = k_t * omega**2
            total_thrust = np.sum(F)

            # Moments (using mixing matrix for '+' configuration)
            # tau_phi = L * (F1 - F3)  [roll]
            # tau_theta = L * (F2 - F4) [pitch]
            # tau_psi = k_d * (-omega1^2 + omega2^2 - omega3^2 + omega4^2) [yaw]

            tau_phi = L * (F[0] - F[2])
            tau_theta = L * (F[1] - F[3])
            tau_psi = k_d * (
                -omega[0] ** 2 + omega[1] ** 2 - omega[2] ** 2 + omega[3] ** 2
            )

            moments = np.array([tau_phi, tau_theta, tau_psi])
        else:
            # Use direct thrust and moment commands
            total_thrust = control.thrust
            moments = np.array([control.tau_phi, control.tau_theta, control.tau_psi])

        return total_thrust, moments

    def _dynamics(self, state_vec: np.ndarray, control: ControlInput) -> np.ndarray:
        """
        Compute state derivatives (equations of motion).

        Newton-Euler Equations:

        Translational dynamics (world frame):
            m * [x_ddot, y_ddot, z_ddot]^T = R * [0, 0, -T]^T + [0, 0, m*g]^T - D*v

        Rotational dynamics (body frame):
            I * [p_dot, q_dot, r_dot]^T = tau - omega × (I * omega)

        Where:
            m = mass
            R = rotation matrix (body to world)
            T = total thrust
            g = gravity
            D = drag coefficient matrix
            I = inertia tensor
            tau = external moments
            omega = angular velocity [p, q, r]

        Args:
            state_vec: Current state as numpy array [x,y,z,vx,vy,vz,phi,theta,psi,p,q,r]
            control: Control input

        Returns:
            State derivative vector
        """
        # Extract state variables
        x, y, z = state_vec[0:3]
        vx, vy, vz = state_vec[3:6]
        phi, theta, psi = state_vec[6:9]
        p, q, r = state_vec[9:12]

        # Get parameters
        m = self.params.mass
        g = self.params.gravity
        Ixx, Iyy, Izz = self.params.Ixx, self.params.Iyy, self.params.Izz

        # Compute thrust and moments from control input
        thrust, moments = self._compute_thrust_and_moments(control)

        # Rotation matrix (body to world)
        R = self._rotation_matrix(phi, theta, psi)

        # Thrust vector in body frame (points up, so negative Z in NED)
        thrust_body = np.array([0, 0, -thrust])

        # Thrust in world frame
        thrust_world = R @ thrust_body

        # Gravity vector in world frame (points down in NED)
        gravity_world = np.array([0, 0, m * g])

        # Aerodynamic drag (proportional to velocity squared, opposing motion)
        v_world = np.array([vx, vy, vz])
        v_mag = np.linalg.norm(v_world)
        if v_mag > 0.01:
            drag = (
                -0.5
                * self.params.air_density
                * np.array(
                    [
                        self.params.drag_coeff_x,
                        self.params.drag_coeff_y,
                        self.params.drag_coeff_z,
                    ]
                )
                * v_world
                * v_mag
            )
        else:
            drag = np.zeros(3)

        # Translational acceleration (Newton's second law)
        # m * a = F_thrust + F_gravity + F_drag
        accel = (thrust_world + gravity_world + drag) / m

        # Angular velocity vector
        omega_vec = np.array([p, q, r])

        # Gyroscopic moments: omega × (I * omega)
        I_omega = np.array([Ixx * p, Iyy * q, Izz * r])
        gyroscopic = np.cross(omega_vec, I_omega)

        # Angular acceleration (Euler's equation for rigid body)
        # I * omega_dot = tau - omega × (I * omega)
        angular_accel = np.array(
            [
                (moments[0] - gyroscopic[0]) / Ixx,
                (moments[1] - gyroscopic[1]) / Iyy,
                (moments[2] - gyroscopic[2]) / Izz,
            ]
        )

        # Euler angle rates
        T = self._euler_rate_matrix(phi, theta)
        euler_rates = T @ omega_vec

        # Assemble state derivative
        state_dot = np.array(
            [
                vx,
                vy,
                vz,  # Position derivatives = velocity
                accel[0],
                accel[1],
                accel[2],  # Velocity derivatives = acceleration
                euler_rates[0],
                euler_rates[1],
                euler_rates[2],  # Euler angle derivatives
                angular_accel[0],
                angular_accel[1],
                angular_accel[2],  # Angular accel
            ]
        )

        return state_dot

    def _euler_integration(
        self, state_vec: np.ndarray, control: ControlInput, dt: float
    ) -> np.ndarray:
        """
        Euler method numerical integration.

        x(t + dt) = x(t) + dt * f(x(t), u(t))

        Simple first-order method. Fast but may accumulate error
        and can be unstable for large time steps.

        Args:
            state_vec: Current state
            control: Control input
            dt: Time step [s]

        Returns:
            New state after integration
        """
        k1 = self._dynamics(state_vec, control)
        return state_vec + dt * k1

    def _rk4_integration(
        self, state_vec: np.ndarray, control: ControlInput, dt: float
    ) -> np.ndarray:
        """
        4th-order Runge-Kutta numerical integration.

        Classic RK4 method with 4 evaluations per step:
        k1 = f(x, u)
        k2 = f(x + dt/2 * k1, u)
        k3 = f(x + dt/2 * k2, u)
        k4 = f(x + dt * k3, u)
        x_new = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        Fourth-order accurate (error ~ O(dt^5)), more stable than Euler.

        Args:
            state_vec: Current state
            control: Control input
            dt: Time step [s]

        Returns:
            New state after integration
        """
        k1 = self._dynamics(state_vec, control)
        k2 = self._dynamics(state_vec + 0.5 * dt * k1, control)
        k3 = self._dynamics(state_vec + 0.5 * dt * k2, control)
        k4 = self._dynamics(state_vec + dt * k3, control)

        return state_vec + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def step(self, control: ControlInput, dt: float) -> QuadcopterState:
        """
        Advance the simulation by one time step.

        This is the main simulation update function. It:
        1. Performs numerical integration of the equations of motion
        2. Updates the state
        3. Records state history for visualization

        Args:
            control: Control input for this time step
            dt: Time step [s]

        Returns:
            Updated quadcopter state
        """
        # Get current state as array
        state_vec = self.state.to_array()

        # Integrate dynamics
        if self.integration_method == IntegrationMethod.EULER:
            new_state_vec = self._euler_integration(state_vec, control, dt)
        else:  # RK4
            new_state_vec = self._rk4_integration(state_vec, control, dt)

        # Update timestamp
        new_timestamp = self.state.timestamp + dt

        # Update rotor speeds (first-order lag model for motor dynamics)
        if control.use_rotor_speeds:
            alpha = dt / (self.params.motor_time_constant + dt)
            new_omega = self.state.omega + alpha * (
                control.omega_cmd - self.state.omega
            )
        else:
            new_omega = self.state.omega.copy()

        # Create new state
        self.state = QuadcopterState.from_array(new_state_vec, new_omega, new_timestamp)

        # Update history
        self._update_history()

        # Update statistics
        self.total_steps += 1
        self.total_simulation_time += dt

        return self.state

    def _update_history(self):
        """Update position and velocity history buffers."""
        pos = self.state.get_position().tolist()
        vel = self.state.get_velocity().tolist()

        self.position_history.append(pos)
        self.velocity_history.append(vel)

        # Keep only last N values
        if len(self.position_history) > self.max_history_length:
            self.position_history.pop(0)
        if len(self.velocity_history) > self.max_history_length:
            self.velocity_history.pop(0)

    def get_state(self) -> QuadcopterState:
        """Return current state."""
        return self.state

    def get_position_history(self) -> list:
        """Return last 10 position values for visualization."""
        return self.position_history.copy()

    def get_velocity_history(self) -> list:
        """Return last 10 velocity values for visualization."""
        return self.velocity_history.copy()

    def set_integration_method(self, method: IntegrationMethod):
        """Change the integration method at runtime."""
        self.integration_method = method

    def set_parameters(self, params: QuadcopterParameters):
        """Update quadcopter parameters at runtime."""
        self.params = params

    def get_hover_thrust(self) -> float:
        """Calculate the thrust required for hover (equilibrium)."""
        return self.params.mass * self.params.gravity

    def compute_stability_metrics(self) -> dict:
        """
        Compute stability metrics for the current state.

        Returns dictionary with:
        - kinetic_energy: Total kinetic energy [J]
        - potential_energy: Gravitational potential energy [J]
        - total_energy: Sum of kinetic and potential [J]
        - angular_momentum: Angular momentum magnitude [kg·m²/s]
        """
        m = self.params.mass
        v = self.state.get_velocity()
        omega = self.state.get_angular_velocity()
        h = -self.state.z  # Height above ground (positive up)

        # Kinetic energy (translational + rotational)
        KE_trans = 0.5 * m * np.dot(v, v)
        I = self.params.get_inertia_matrix()
        KE_rot = 0.5 * omega @ I @ omega
        KE = KE_trans + KE_rot

        # Potential energy
        PE = m * self.params.gravity * h

        # Angular momentum magnitude
        L = np.linalg.norm(I @ omega)

        return {
            "kinetic_energy": KE,
            "potential_energy": PE,
            "total_energy": KE + PE,
            "angular_momentum": L,
        }


# Utility functions for testing and demonstration
def create_hover_control(model: QuadcopterModel) -> ControlInput:
    """Create control input for hovering at current position."""
    hover_thrust = model.get_hover_thrust()
    return ControlInput(thrust=hover_thrust, use_rotor_speeds=False)


def simulate_step_response(
    model: QuadcopterModel,
    target_altitude: float = 5.0,
    duration: float = 10.0,
    dt: float = 0.02,
) -> list:
    """
    Simulate a step response in altitude.

    Args:
        model: Quadcopter model instance
        target_altitude: Target altitude in meters (positive up)
        duration: Simulation duration in seconds
        dt: Time step in seconds

    Returns:
        List of states over time
    """
    model.reset()
    states = []

    hover_thrust = model.get_hover_thrust()

    n_steps = int(duration / dt)
    for i in range(n_steps):
        # Simple proportional altitude controller
        current_alt = -model.state.z
        alt_error = target_altitude - current_alt
        thrust = hover_thrust + 2.0 * alt_error  # P-control

        control = ControlInput(thrust=max(0, thrust), use_rotor_speeds=False)
        state = model.step(control, dt)
        states.append(state.copy())

    return states


if __name__ == "__main__":
    # Test the model
    print("Testing Quadcopter Mathematical Model")
    print("=" * 50)

    # Create model with default parameters
    model = QuadcopterModel(integration_method=IntegrationMethod.RUNGE_KUTTA_4)

    print(f"Integration method: {model.integration_method.value}")
    print(f"Hover thrust: {model.get_hover_thrust():.2f} N")
    print(f"Mass: {model.params.mass} kg")
    print(f"Gravity: {model.params.gravity} m/s²")

    # Test hover
    print("\nSimulating hover for 1 second...")
    hover_control = create_hover_control(model)

    for i in range(50):  # 50 steps at 20ms = 1 second
        model.step(hover_control, 0.02)

    state = model.get_state()
    print(f"Position after hover: ({state.x:.4f}, {state.y:.4f}, {state.z:.4f})")
    print(f"Velocity: ({state.vx:.4f}, {state.vy:.4f}, {state.vz:.4f})")

    # Test position history
    print(f"\nPosition history (last {len(model.get_position_history())} values):")
    for i, pos in enumerate(model.get_position_history()):
        print(f"  {i+1}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

    print("\nModel test complete!")
