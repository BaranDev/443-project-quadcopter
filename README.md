# Quadcopter Simulator - CMSE443 Real-Time Systems Design

A comprehensive quadcopter flight simulator with real-time control, PID stabilization, and 3D visualization using Cosys-AirSim. This project demonstrates real-time systems design principles including fixed-rate control loops, timing analysis, and cascaded PID control.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)

## Features

### Core Functionality
- **Real-Time Control System**: Fixed-rate control loop (50Hz) with deadline monitoring
- **Cascaded PID Control**: Position → Velocity → Attitude → Rate control architecture
- **Multiple Input Methods**: Xbox Wireless Controller and keyboard support
- **3D Visualization**: Real-time telemetry display with attitude indicators
- **Mathematical Model**: Newton-Euler equations with RK4 numerical integration
- **Timing Analysis**: WCET (Worst-Case Execution Time) tracking and jitter analysis

### Control Features
- Position hold and waypoint navigation
- Automated takeoff and landing
- Emergency stop functionality
- Geofencing and safety limits
- Real-time parameter tuning via GUI

### Visualization
- Artificial horizon (attitude indicator)
- Heading indicator (compass)
- Top-down position plot with trajectory
- Real-time telemetry graphs
- PID component visualization

## Requirements

### Software Dependencies
- **Python**: 3.8 or higher
- **Cosys-AirSim**: Unreal Engine-based flight simulator
- **Operating System**: Windows 10/11 (for Xbox controller support)

### Python Packages
```bash
pip install -r requirements.txt
```

Required packages:
- `numpy>=1.21.0` - Numerical computations
- `XInput-Python>=0.4.0` - Xbox controller support (Windows)
- `keyboard>=0.13.5` - Keyboard input fallback
- `pytest>=7.0.0` - Testing framework
- `pytest-cov>=4.0.0` - Code coverage

## Getting Started

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/barandev/443-project-quadcopter.git
   cd 443-project-quadcopter
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Cosys-AirSim**
   - Download and install Cosys-AirSim from [official website](https://cosys-lab.github.io/Cosys-AirSim/)
   - Launch the Blocks environment or your custom Unreal environment

### Running the Simulator

1. **Start Cosys-AirSim** and load the Blocks environment

2. **Run the main controller**
   ```bash
   python main.py
   ```

3. **Control the quadcopter**
   - **Xbox Controller** (recommended):
     - Left stick: Pitch/Roll
     - Right stick: Yaw
     - Right trigger: Altitude up
     - Left trigger: Altitude down
     - A button: Arm/Disarm
     - B button: Emergency stop
     - Start: Takeoff
     - Back: Land
   
   - **Keyboard** (fallback):
     - W/S: Pitch forward/backward
     - A/D: Roll left/right
     - Q/E: Yaw left/right
     - Space: Altitude up
     - Left Shift: Altitude down
     - Enter: Arm/Disarm
     - Esc: Emergency stop

## Project Structure

```
443-project-quadcopter/
├── main.py                  # Main controller application
├── quadcopter_model.py      # Mathematical model (Newton-Euler dynamics)
├── pid_controller.py        # PID control implementation
├── input_handler.py         # Xbox controller and keyboard input
├── visualization.py         # Real-time GUI visualization
├── test_suite.py           # Comprehensive test suite
├── test_drone.py           # Basic drone tests
├── requirements.txt        # Python dependencies
├── docs/                   # Documentation and reports
│   ├── CMSE443_Report.txt
│   └── ...
└── Blocks/                 # Cosys-AirSim Unreal project
```

## Control Architecture

### Cascaded PID Control
```
Position Setpoint → [Position PID] → Velocity Command
                                    ↓
                          [Velocity PID] → Attitude Command
                                          ↓
                                [Attitude PID] → Rate Command
                                                ↓
                                      [Rate PID] → Motor Commands
```

### Control Loop Timing
- **Control Rate**: 50 Hz (20ms period)
- **GUI Update Rate**: 20 Hz (50ms period)
- **Deadline Monitoring**: Tracks WCET and deadline misses
- **Integration Method**: 4th-order Runge-Kutta (RK4)

## Testing

Run the comprehensive test suite:
```bash
pytest test_suite.py -v
```

Run with coverage:
```bash
pytest test_suite.py --cov=. --cov-report=html
```

### Test Coverage
- Quadcopter dynamics model
- PID controller functionality
- Input handling and deadzone
- Numerical integration methods
- Safety limits and geofencing

## Performance Metrics

The simulator tracks and displays:
- **Loop Time**: Actual control loop execution time
- **WCET**: Worst-case execution time
- **Jitter**: Variation in loop timing
- **Deadline Misses**: Number of missed control deadlines
- **PID Components**: P, I, D terms for each controller

## Configuration

### PID Tuning
PID gains can be adjusted in real-time via the GUI or by modifying the controller initialization in `pid_controller.py`:

```python
# Position control gains
position_gains = PIDGains(Kp=1.0, Ki=0.1, Kd=0.5)

# Attitude control gains
attitude_gains = PIDGains(Kp=5.0, Ki=0.5, Kd=1.0)

# Rate control gains
rate_gains = PIDGains(Kp=0.8, Ki=0.2, Kd=0.1)
```

### Safety Limits
Configure safety parameters in `main.py`:
```python
config = ControllerConfig(
    max_velocity_xy=5.0,      # m/s
    max_velocity_z=3.0,       # m/s
    max_altitude=50.0,        # m
    geofence_radius=100.0     # m
)
```

## Technical Details

### Mathematical Model
The quadcopter dynamics are modeled using:
- **Newton-Euler equations** for rigid body motion
- **NED coordinate frame** (North-East-Down)
- **Euler angles** (roll, pitch, yaw) for orientation
- **Aerodynamic effects**: Drag and rotor dynamics

### Numerical Integration
- **Euler method**: Fast, first-order accuracy
- **RK4 method**: Higher accuracy, fourth-order (default)

### Input Processing
- **Deadzone compensation**: Eliminates stick drift
- **Exponential response curves**: Finer control at low inputs
- **Smoothing filters**: Reduces input noise
- **Haptic feedback**: Controller vibration for events

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**barandev**
- GitHub: [@barandev](https://github.com/barandev)

## Acknowledgments

- **CMSE443 Real-Time Systems Design** course
- **Cosys-AirSim** team for the excellent simulation platform
- References:
  - Beard, R. W., & McLain, T. W. (2012). *Small unmanned aircraft: Theory and practice*
  - Bouabdallah, S. (2007). *Design and control of quadrotors with application to autonomous flying*

## Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/barandev/443-project-quadcopter/issues) page
2. Create a new issue with detailed information
3. Include error messages and system information

## Roadmap

- [ ] Add autonomous waypoint navigation
- [ ] Implement obstacle avoidance
- [ ] Add support for custom Unreal environments
- [ ] Multi-drone simulation support
- [ ] Advanced flight modes (acrobatic, racing)
- [ ] Data logging and replay functionality

---

**Note**: This project was developed as part of the CMSE443 Real-Time Systems Design course. It demonstrates practical implementation of real-time control systems, PID controllers, and software integration with physics simulators.
