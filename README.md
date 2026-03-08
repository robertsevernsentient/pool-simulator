# Pool Simulator

## Overview
The Pool Simulator is a Python application that simulates the physics of a pool game. It provides a realistic environment for users to interact with virtual pool balls on a pool table, allowing for the exploration of dynamics, collisions, and forces.

## Features
- Realistic physics calculations for ball movement and interactions.
- User-friendly interface for simulating pool games.
- Modular design with separate components for physics, geometry, and utilities.

## Project Structure
```
pool-simulator
├── src
│   ├── pool_simulator
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── simulator.py
│   │   ├── physics
│   │   │   ├── __init__.py
│   │   │   ├── dynamics.py
│   │   │   ├── collisions.py
│   │   │   └── forces.py
│   │   ├── geometry
│   │   │   ├── __init__.py
│   │   │   ├── ball.py
│   │   │   └── table.py
│   │   └── utils
│   │       ├── __init__.py
│   │       └── helpers.py
│   └── tests
│       ├── __init__.py
│       ├── test_dynamics.py
│       └── test_collisions.py
├── pyproject.toml
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation
To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd pool-simulator
pip install -r requirements.txt
```

## Usage
To run the pool simulator, execute the following command:

```bash
python src/pool_simulator/main.py
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.