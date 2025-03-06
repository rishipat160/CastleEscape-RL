Author: Rishi Patel

This project is done as a part of the course CS5100 at Northeastern University.

# Castle Escape - Reinforcement Learning

A reinforcement learning project implementing various algorithms to solve the Castle Escape game environment.

## Project Overview

This project implements and compares different reinforcement learning approaches to solve a castle escape game. The player must navigate through a castle, avoid or defeat guards, and reach the goal room while maintaining health.

## Environment Description

The Castle Escape environment is a 5x5 grid world where:
- The player starts at position (0,0)
- The goal is at position (4,4)
- Four guards are randomly placed in the grid
- The player has three health states: Full (2), Injured (1), and Critical (0)
- The game ends when the player reaches the goal or health becomes Critical

### Actions
- Movement: UP, DOWN, LEFT, RIGHT
- Combat: FIGHT
- Stealth: HIDE

### State Space
Each state is defined by:
- Player position (x,y)
- Player health (0-2)
- Guard positions

### Rewards
- Reaching the goal: +10000
- Winning combat: +10
- Losing combat: -1000
- Defeat (health becomes Critical): -1000

## Implemented Algorithms

### Model-Based Monte Carlo (MBMC)
The `MBMC.py` file implements a model-based Monte Carlo approach to estimate the probability of victory against each guard when taking the fight action.

### Model-Free Monte Carlo (MFMC)
The `MFMC.py` file implements Q-learning, a model-free reinforcement learning algorithm, to learn an optimal policy for navigating the castle.

## Usage

To run the algorithms:
Run the Model-Based Monte Carlo approach
``` bash
python MBMC.py
```

Run the Model-Free Monte Carlo (Q-learning) approach
``` bash
python MFMC.py
```

To enable visualization, set `gui_flag = True` at the top of each file.

## Visualization

The project includes a visualization module (`vis_gym.py`) that provides a graphical interface for the game environment. When enabled, it shows:
- The player (green circle)
- Guards (red squares)
- The goal room (yellow)
- Player health status
- Action results in a console area

## Files

- `mdp_gym.py`: Defines the Castle Escape environment as a Gym environment
- `vis_gym.py`: Visualization module for the environment
- `MBMC.py`: Model-Based Monte Carlo implementation
- `MFMC.py`: Model-Free Monte Carlo (Q-learning) implementation

## Requirements

- Python 3.x
- NumPy
- Pygame (for visualization)
- OpenAI Gym

## Author

Rishi Patel

*This project was completed as part of the course CS5100 at Northeastern University.*