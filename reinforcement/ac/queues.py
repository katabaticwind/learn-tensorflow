from heapq import heappush, heappush, heappop, heapify, heapreplace
from collections import deque
import numpy as np
import time

class Queue():
    """Basic FIFO queue."""

    def __init__(self, size, init_values=[]):
        assert len(init_values) <= size, "Initial values exceed queue length."
        self.size = size
        self.values = deque(init_values)

    def __len__(self):
        return len(self.values)

    def push(self, value):
        """Add a memory to queue."""
        if len(self.values) == self.size:
            self.values.popleft()
            self.values.append(value)
        else:
            self.values.append(value)

    def fill(self, min_len):
        """Fill queue up to `min_len` by copying the first element."""
        while len(self) < min_len:
            self.values.appendleft(self.values[0])

    def get(self, index):
        return self.values[index]

class PriorityQueue():
    """Implementation of priority queue.

        In prioritized replay, we add new memories with priority N (highest priority), and remove a memory in the lowest priority (we can't guaruntee to remove the lowest priority memory because the heap queue isn't sort stable, i.e. the last memory isn't necessarily the lowest priority memory, but the first memory is).

        Every 1e6 steps we perform a full sort of the heap queue.

        Heap queue is a "min" queue, so we use *negative* TD-errors as sorting criteria.

        Using a priority queue requires following changes to training:

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
            add_memory(replay_memory, (-np.inf, -global_steps, memory))  # stored with maximal priority
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

    def __init__(self, env, size, alpha, beta, segments):
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.queue = []
        self.endpoints = []  # right-hand endpoints of equally-probability segments
        self.fill_queue(env)
        self.calculate_total_priority()
        self.find_endpoints(segments)

    def fill_queue(self, env):
        """Create replay memory by performing random actions in environment.

            Initial transitions are given priority zero (lowest possible priority).

            Queue items have form (-td_error, -step, memory), where `td_error` is the *absolute* TD error.

            The error and step are negative because `heapq` implements a min. binary queue, meaning that *smaller* values are added to the beginning of the queue. Therefore, a new observation assigned an error of `-np.inf` and `-global_step` will *always* be assigned to the beginning of the queue, as it has the smallest possible error, and the smallest step up to that point.
        """
        print("Filling replay memory queue... ", end="")
        steps = 0
        while steps < self.size:
            state = env.reset()
            while True:
                action = env.action_space.sample()
                next_state, reward, done, info = env.step(action)
                steps += 1
                memory = [state, action, reward, next_state, done]
                heappush(self.queue, [0, -steps, memory])
                state = next_state
                if done or steps == self.size:
                    break
        print("done.")

    def find_endpoints(self, batch_size):
        """Find the endpoints of `batch_size` equal-probability segments."""
        # print("Determining replay memory segments... ", end="")
        self.endpoints = []
        s = 0.0
        segment = 1
        for i in np.arange(1, self.size):  # don't hit last idx => never s > segment / batch_size
            s += (1 / i) ** self.alpha / self.total_priority
            if s > segment / batch_size:
                self.endpoints.append(i)
                segment += 1
        self.endpoints += [self.size]
        # print("done.")

    def set_alpha(self, alpha):
        self.alpha = alpha
        self.calculate_total_priority()
        # print(len(self.endpoints))
        self.find_endpoints(len(self.endpoints))

    def set_beta(self, beta):
        self.beta = beta

    def add_memory(self, memory, step):
        self.queue.pop()  # remove a low priority memory
        heappush(self.queue, [-np.inf, -step, memory])  # add new memory at highest priority

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
        t0 = time.time()
        print("Sorting replay memory... ", end="")
        self.queue.sort()  # low to high
        t1 = time.time()
        print("done (elapsed time: {:.2f} s).".format(t1 - t0))

    def sample_queue(self):
        """Sample `batch_size` memories from replay memory, where `batch_size` is determined by the length of `self.endpoints`."""
        idx = np.random.randint(0, self.endpoints[0])
        sample_idx = [idx]
        sample = [self.queue[idx][2]]
        for i in np.arange(1, len(self.endpoints)):
            # if self.endpoints[i - 1] >= self.endpoints[i]:
            #     print(self.endpoints[i - 1], self.endpoints[i])
            idx = np.random.randint(self.endpoints[i - 1], self.endpoints[i])
            sample_idx.append(idx)
            sample.append(self.queue[idx][2])
        return sample_idx, sample

    def calculate_total_priority(self):
        """Find the total priority based on the ranking methodology.

            The priority of the `i`-th memory in the priority queue is `p[i] = (1 / i) ** alpha`.

            The total priority only depends on the size of the queue and `alpha`, so this function only needs to be called if one of these values changes (`N` is generally fixed, but `alpha` typically has a pre-determined schedule).
        """

        # print("Setting total replay memory priority... ", end="")
        self.total_priority = np.sum(1 / np.arange(1, self.size + 1) ** self.alpha)
        # print("done.")

    def calculate_probability(self, idx):
        """Calculate the probability associated with memory at index `idx` of queue."""
        return (1 / idx) ** self.alpha / self.total_priority

    def calculate_weight(self, idx):
        """Find the importance-sampling weight of the `idx`-th memory in the priority queue.

            # Example
            normalization = calculate_total_priority(size, alpha)
            weight = calculate_importance_sampling_weight(idx, size, beta, normalization)
        """
        p_min = self.calculate_probability(self.size)  # max weight attained by min probability (approx. last mem. in queue)
        w_max = (self.size * p_min) ** -self.beta
        p = self.calculate_probability(idx)
        return (self.size * p) ** -self.beta / w_max

def unpack_batch(batch):
    states = np.array([b[0] for b in batch])
    actions = np.array([b[1] for b in batch])
    rewards = np.array([b[2] for b in batch])
    next_states = np.array([b[3] for b in batch])
    dones = np.array([b[4] for b in batch])
    return states, actions, rewards, next_states, dones
