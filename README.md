# Group Project Setup Guide

## Project Content

* Gymnasium v1.2.2
* Part 1: MountainCar
* Part 2: FrozenLake
* Part 3: OOP Custom Environment (Warehouse Robot)

---

## Installation

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

```bash
# 3. Navigate to the Gymnasium directory
cd group_project/Gymnasium

# 4. Install Gymnasium in editable mode
pip install -e .

# 5. Install additional dependencies
pip install "gymnasium[classic_control]"
pip install matplotlib
pip install pygame
```

---

## Verification

Run the following command to verify that the installation is successful:

```bash
pip list
```

Sample Output (MacOS):

```
Package              Version Editable project location
-------------------- ------- --------------------------------------------
cloudpickle          3.1.2
Farama-Notifications 0.0.4
gymnasium            1.2.2   ./group_project/Gymnasium
numpy                2.3.5
pip                  24.3.1
typing_extensions    4.15.0
```

If your output is similar, your environment is correctly configured.

---

## Running the Project

### Part 1: Mountain Car

Train the agent:

```bash
python mountain_car.py --train --episodes 5000
```

Render and visualize performance:

```bash
python mountain_car.py --render --episodes 10
```

Outputs:

* `mountain_car.pkl` (saved Q-table)
* `mountain_car.png` (learning curve)

---

### Part 2: Frozen Lake

Run training and evaluation:

```bash
python frozen_lake.py
```

Details:

* Training: 15,000 episodes
* Evaluation: 1,000 episodes
* Success is defined as reaching the goal (`reward == 1.0`)
* Final success rate is printed in the console
* Learning curve saved as `frozen_lake8x8.png`

---

### Part 3: OOP Project Environment (Warehouse Robot)

Run the OOP-based custom Gym environment:

```bash
python main_part3.py
```

The program allows you to:

* Choose the environment (Basic or Advanced Warehouse)
* Choose the agent (RandomAgent or GreedyAgent)
* Run statistics, visual demo, or both

This part demonstrates:

* Abstraction
* Inheritance
* Polymorphism
* Encapsulation

---

## Notes

* Python 3 is required
* Gymnasium v1.2.2 is used
* Rendering in Part 3 uses Pygame
* This project is for academic use only

---
