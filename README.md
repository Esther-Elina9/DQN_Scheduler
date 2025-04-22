# Research on Cloud-Edge Load Balancing and Task Scheduling Strategies Based on Deep Reinforcement Learning

## Introduction

This project implements an intelligent resource scheduling algorithm based on an improved Dueling DQN for optimizing virtual machine resource configuration in cloud edge collaborative environments.
The algorithm achieves adaptive resource management for dynamic workloads through mechanisms such as integrated experience replay and dual network architecture.
##Characteristics

- **Environmental modeling**:

  - Support independent resource pools for multiple virtual machines
  - Random task generation (normal distribution resource requirements)
  - Dynamic queue management system
  - Multi dimensional state representation (CPU/RAM utilization, queue length, etc.)

- **Algorithm improvement**:

  - Dueling DQN Network Architecture
  - Priority Experience Replay (PER) mechanism
  - Design of Multi Objective Reward Function
  - Dual network asynchronous update strategy

- **Training Monitoring**:
  - Real time recording of key indicators (resource utilization, queue length, etc.)
  - TD error dynamic tracking
  - Automatic saving of model weights
  - Visualization support for training process

## Environment

- Python 3.8.20
- Tensorflow 2.13.0
- NumPy 1.24.3
- Matplotlib 3.7.5
- Keras 2.13.1
  ##Installation

## Quick start

1. Clone this project:

```bash
git clone https://github.com/Esther-Elina9/DQN_Scheduler.git
```

2. Install dependencies:

```bash
pip install tensorflow numpy
```

```bash
pip install matplotlib
```

```bash
pip install unittest
```

3. Enter project directory:

```bash
cd DQN_Scheduler
```

4. Run Project:

- **Training**:

```bash
python DQN2.py
```

The training process will be saved in "training_history.npz".
The training weight will be saved in "dqn_model_weights.h5".

- **Visualization results**:

```bash
python results.py
```

or(Only training curves without printed results)

```bash
python Visualization.py
```

- **Test cases**:

```bash
python -m unittest test_dqn.py -v
```

## Contact Information

If you have any questions, please contact us through the following methods:

- GitHub Issues: https://github.com/Esther-Elina9/DQN_Scheduler/issues
- Email: esther_elina@outlook.com
