# Defaults
Default settings for various reinforcement learning algorithms and tasks.

## CartPole-v0
- ### `pg-baseline`
  - hidden_units: 32
  - batch_size: 5000
  - learning_rate: 1e-2
- ### `dqn-vanilla`
  - hidden_units: 64
  - learning_rate: 1e-3

## LunarLander-v2
- ### `pg-baseline`
  - hidden_units: 64, 64
  - learning_rate: 1e-3
  - batches: 300
- ### `dqn-vanilla`
