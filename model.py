import torch.nn as nn


class ActorNet(nn.Module):
    def __init__(self, in_dim, out_dim, device):
        super(ActorNet, self).__init__()
        self.device = device
        self.net = nn.Sequential(nn.Linear(in_dim, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                                 nn.Linear(512, out_dim), nn.Tanh())  # +-1 output

    def forward(self, state):
        """
        :param state: N, in_dim
        :return: Action (deterministic), N,out_dim
        """
        action = self.net(state)
        return action


class CriticNet(nn.Module):
    def __init__(self, s_dim, a_dim, device):
        super(CriticNet, self).__init__()
        self.device = device
        in_dim = s_dim + a_dim
        self.net = nn.Sequential(nn.Linear(in_dim, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                                 nn.Linear(512, 1))

    def forward(self, sa_pairs):
        """
        :param sa_pairs: state-action pairs, (N, in_dim)
        :return: Q-values , N,1
        """
        return self.net(sa_pairs)
