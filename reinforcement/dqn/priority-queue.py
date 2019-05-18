from heapq import heappush, heappush, heappop, heapify, heapreplace
import numpy as np

class PriorityQueue():
    """Implementation of priority queue.

        In prioritized replay, we add new memories with priority N (highest priority), and remove a memory in the lowest priority (we can't guaruntee to remove the lowest priority memory because the heap queue isn't sort stable, i.e. the last memory isn't necessarily the lowest priority memory, but the first memory is).

        Every 1e6 steps we perform a full sort of the heap queue.

        Heap queue is a "min" queue, so we use *negative* TD-errors as sorting criteria.
    """

    def __init__(self, size, alpha, beta):
        super(PriorityQueue, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.queue = []
        self.endpoints = []  # right-hand endpoints of equally-probability segments

    def fill_queue(self, env):
        """Create replay memory by performing random actions in environment.

            Initial transitions are given priority zero (lowest possible priority).

            Queue items have form (-td_error, -step, memory), where `td_error` is the *absolute* TD error.

            The error and step are negative because `heapq` implements a min. binary queue, meaning that *smaller* values are added to the beginning of the queue. Therefore, a new observation assigned an error of `-np.inf` and `-global_step` will *always* be assigned to the beginning of the queue, as it has the smallest possible error, and the smallest step up to that point.
        """
        # replay_memory = []
        steps = 0
        while steps < self.size:
            state = env.reset()
            while True:
                action = env.action_space.sample()
                next_state, reward, done, info = env.step(action)
                steps += 1
                memory = [state, action, reward, next_state, done]
                heappush(self.queue, [0, steps, memory])
                state = next_state
                if done or steps == self.size:
                    break

    def find_endpoints(self, batch_size):
        """Find the endpoints of `batch_size` equal-probability segments."""
        self.endpoints = []
        normalization = np.sum(1 / np.arange(1, self.size + 1) ** self.alpha)
        s = 0.0
        segment = 1
        for i in np.arange(1, self.size + 1):
            s += (1 / i) ** self.alpha / normalization
            if s > segment / batch_size:
                self.endpoints.append(i)
                segment += 1
        self.endpoints += [self.size]

    def add_memory(self, memory, step):
        self.queue.pop()  # remove a low priority memory
        heappush(self.queue, (-np.inf, step, memory))  # add new memory at highest priority

    def remove_memory(self, memory_idx):
        memory = self.queue.pop(memory_idx)
        return memory

    def update_memory(self, memory_idx, priority):
        """Change the priority of an existing memory in the queue.

            First remove the memory, then add it back with new priority.
        """
        memory = self.remove_memory(memory_idx)
        memory[0] = priority
        heappush(self.queue, memory)

    def sort_queue(self):
        self.queue.sort()  # low to high

    def sample_queue(self):
        """Sample `batch_size` memories from replay memory, where `batch_size` is determined by the length of `self.endpoints`."""
        idx = np.random.randint(0, self.endpoints[0])
        sample_idx = [idx]
        sample = [self.queue[idx][2]]
        for i in np.arange(1, len(self.endpoints)):
            idx = np.random.randint(self.endpoints[i - 1], self.endpoints[i])
            sample_idx.append(idx)
            sample.append(self.queue[idx][2])
        return sample_idx, sample

    def calculate_total_priority(self):
        """Find the total priority based on the ranking methodology.

            The priority of the `i`-th memory in the priority queue is `p[i] = (1 / i) ** alpha`.

            The total priority only depends on the size of the queue and `alpha`, so this function only needs to be called if one of these values changes (`N` is generally fixed, but `alpha` typically has a pre-determined schedule).
        """

        self.total_priority = np.sum(1 / np.arange(1, self.size + 1) ** self.alpha)

    def calculate_probability(self, idx):
        """Calculate the probability associated with memory at index `idx` of queue."""
        return (1 / idx) ** self.alpha / self.total_priority

    def calculate_importance_sampling_weight(self, idx):
        """Find the importance-sampling weight of the `idx`-th memory in the priority queue.

            # Example
            normalization = calculate_total_priority(size, alpha)
            weight = calculate_importance_sampling_weight(idx, size, beta, normalization)
        """
        p_min = self.calculate_probability(self.size, self.alpha, self.total_priority)  # max weight attained by min probability
        w_max = (self.size * p_min) ** -self.beta
        p = self.calculate_probability(idx, self.alpha, self.total_priority)
        return (self.size * p) ** -self.beta / w_max

"""
# Changes to the graph
weights_pl = tf.placeholder(dtype=tf.float32, shape=(None,))

errors = tf.math.abs(target_values - estimates)
loss = tf.losses.mean_squared_error(targets, estimates, weights=weights_pl)

# Changes to the episode
state = env.reset()
while True:
    action = ...  # e.g. env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    memory = [state, action, reward, next_state, done]
    add_memory(replay_memory, (-np.inf, global_steps, memory))  # stored with maximal priority
    if global_steps % update_freq == 0:  # perform an update...
        # sample replay memory
        batch_idx, batch = sample_replay_memory(replay_memory, endpoints)
        # compute importance-sampling weights for batch
        normalization = compute_total_probability(memory_size, memory_alpha)
        weights = [calculate_importance_sampling_weight(memory, idx, size, memory_beta, normalization) for (memory, idx) in zip(batch, batch_idx)]
        # perform update *and* compute TD errors
        td_errors = update(...)  # will call `sess.run([errors, train_op], ...)`
        # update memory priorities
        for (idx, error) in zip(batch_idx, td_errors):
            update_memory(replay_memory, idx, error)
"""
