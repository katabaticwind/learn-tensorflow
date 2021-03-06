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
  - clone_steps: 1000
- ### `dqn-rainbow`
  - hidden_units: 64
  - learning_rate: 1e-2
  - memory_size: 10000
  - clone_steps: 1000
  - sort_freq: 1000

## LunarLander-v2
- ### `pg-baseline`
  - hidden_units: 64, 64
  - learning_rate: 1e-3
  - batches: 300
- ### `dqn-vanilla`
  - hidden_units: 64, 64
  - learning_rate: 1e-3 (1e-4)
  - batch_size: 64
  - lr_decay: 1.00
  - discount_factor: 0.99
  - init_epsilon: 1.00
  - min_epsilon: 0.01
  - eps_decay: 0.995
  - episodes: 3000 (2000)
  - update_freq: 4
  - clone_steps: 1e4
  - min_memory_size: 1e4
  - max_memory_size: 1e5
