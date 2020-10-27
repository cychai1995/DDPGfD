import numpy as np
import torch, operator
from collections import deque


class NStepBackup:
    def __init__(self, gamma, n_step=5):
        self.n_step = n_step  # -1 for monte carlo
        self.buffer = deque()
        self.gamma = gamma  # Discount

    def reset(self):
        del self.buffer
        self.buffer = deque()

    def add_exp(self, exp):
        self.buffer.append(exp)

    def available(self, done):
        return (len(self.buffer) >= self.n_step) or (done and len(self.buffer) > 0)

    def pop_exp(self, done=False):  # Pop experience with n-step return and other meta data
        if len(self.buffer) >= self.n_step:
            s, a, g, s2 = self.buffer.popleft()
            g = g.float()
            step = 1  # record actual step as the power of Q's decay factor
            gamma = self.gamma
            for i in range(min(len(self.buffer), self.n_step - 1)):
                _, _, r, s2 = self.buffer[i]  # a_end for selecting Q
                g += gamma * r
                step += 1
                gamma *= self.gamma
        else:
            if done and len(self.buffer) > 0:  # not maintaining the exact n step
                s, a, g, s2 = self.buffer.popleft()
                g = g.float()
                step = 1  # record actual step as the power of Q's decay factor
                gamma = self.gamma
                for i in range(min(len(self.buffer), self.n_step - 1)):
                    _, _, r, s2 = self.buffer[i]
                    g += gamma * r
                    step += 1
                    gamma *= self.gamma
            else:
                return False, None
        return True, (s, a, torch.tensor([g]), s2, torch.tensor([gamma]))


# https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.
        https://en.wikipedia.org/wiki/Segment_tree
        Can be used as regular array, but with two
        important differences:
            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.
        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements (i.e. be associative)
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences
        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)


class ReplayBuffer(object):
    def __init__(self, size, seed=None):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = np.empty(size, dtype=object)
        self._maxsize = size
        self.cur_sz = 0
        self._next_idx = 0
        self.rs = np.random.RandomState(seed)
        self.protect_idx = -1

    def set_protect_size(self, protect_size): # For keeping demonstration data,keep first protect_size items
        self.protect_idx = protect_size-1

    def __len__(self):
        return self.cur_sz

    def add(self, experience):  # Experience: tuple of (s,a,r,s2) with CPU tensor type
        self._storage[self._next_idx] = experience
        self._next_idx = (self._next_idx + 1) % self._maxsize
        if self._next_idx == 0:
            self._next_idx = self.protect_idx + 1
        self.cur_sz = min(self.cur_sz + 1, self._maxsize)

    def _encode_sample(self, idxes):
        s_, a_, r_, s2_, gamma_, flag_ = [], [], [], [], [], []
        exps = self._storage[idxes]
        for exp in exps:
            s, a, r, s2, gamma, flag = exp
            s_.append(s.clone())
            a_.append(a.clone())
            r_.append(r.clone())
            s2_.append(s2.clone())
            gamma_.append(gamma.clone())
            flag_.append(flag)
        # stack along new axis
        return torch.stack(s_), torch.stack(a_), torch.stack(r_), \
               torch.stack(s2_), torch.stack(gamma_), np.stack(flag_)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        gammas: np.array
            product of gammas for N-step returns
        """
        idxes = self.rs.randint(0, self.cur_sz - 1, batch_size)
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, seed, alpha, beta_init=0.4, beta_inc_n=2000):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size, seed)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

        self.beta_inc = (1 - beta_init) / beta_inc_n
        self.beta = beta_init

    def ready(self):
        return self.cur_sz > 1

    def update_beta(self):
        self.beta = min(1, self.beta + self.beta_inc)

    def add(self, experience):
        idx = self._next_idx
        super().add(experience)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, self.cur_sz - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = self.rs.uniform(0, 1) * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        # assert beta > 0
        batch_size = min(self.cur_sz, batch_size)
        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self.cur_sz) ** (-self.beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * self.cur_sz) ** (-self.beta)
            weights.append(weight / max_weight)
        weights = np.array(weights, dtype=np.float32)
        encoded_sample = self._encode_sample(idxes)
        return encoded_sample, weights, idxes

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < self.cur_sz
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


def test_nstep():
    bk = NStepBackup(0.1, 5)
    reward = np.arange(20)
    for i in range(20):
        bk.add_exp((i, i, i, i))
        # for i in range(9):
        # available, exp = bk.pop_exp()
        # print(available, exp)
    available = True
    while available:
        available, exp = bk.pop_exp()
        print(available, exp)
    # Done version,not exact n step
    available = True
    while available:
        available, exp = bk.pop_exp(done=True)
        print(available, exp)


if __name__ == '__main__':
    test_nstep()
