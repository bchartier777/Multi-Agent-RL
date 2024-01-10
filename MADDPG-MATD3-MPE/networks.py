import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

SIGMA_MIN = -20
SIGMA_MAX = 2

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


# Different agents have different observation dimensions and action dimensions, so we need to use 'agent_id' to distinguish them
class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.max_action = args.max_action
        self.gumb_softmax_action = args.gumb_softmax_action
        self.dim1 = args.obs_dim_n[agent_id]; self.dim2 = args.hidden_dim
        self.fc1 = nn.Linear(self.dim1, self.dim2)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.action_dim_n[agent_id])
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = F.relu(self.fc3(x))
        a = self.max_action * torch.tanh(self.fc3(x))

        if self.gumb_softmax_action == True:
           return a, logits
        else:
           return a

class Critic_MADDPG(nn.Module):
    def __init__(self, args):
        super(Critic_MADDPG, self).__init__()
        self.fc1 = nn.Linear(sum(args.obs_dim_n) + sum(args.action_dim_n), args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s, a):
        s = torch.cat(s, dim=1)
        a = torch.cat(a, dim=1)
        s_a = torch.cat([s, a], dim=1)

        q = F.relu(self.fc1(s_a))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q

class Actor_v2(nn.Module):
    def __init__(self, args, agent_id, use_batch_norm=False,
                 fc1_units=512, fc2_units=256, fc3_units=128, fc4_units=64, fc5_units=32):
        """
        :param observation_size: observation size
        :param action_size: action size
        :param use_batch_norm: True to use batch norm
        :param seed: random seed
        :param fc1_units: number of nodes in 1st hidden layer
        :param fc2_units: number of nodes in 2nd hidden layer
        :param fc3_units: number of nodes in 3rd hidden layer
        :param fc4_units: number of nodes in 4th hidden layer
        :param fc5_units: number of nodes in 5th hidden layer
        """
        super(Actor_v2, self).__init__()

        if args.seed is not None:
            torch.manual_seed(args.seed)

        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(args.obs_dim_n[agent_id])
            self.bn2 = nn.BatchNorm1d(fc1_units)
            self.bn3 = nn.BatchNorm1d(fc2_units)
            self.bn4 = nn.BatchNorm1d(fc3_units)
            self.bn5 = nn.BatchNorm1d(fc4_units)
            self.bn6 = nn.BatchNorm1d(fc5_units)

        # batch norm has bias included, disable linear layer bias
        use_bias = not use_batch_norm

        self.use_batch_norm = use_batch_norm
        self.fc1 = nn.Linear(args.obs_dim_n[agent_id], fc1_units, bias=use_bias)
        self.fc2 = nn.Linear(fc1_units, fc2_units, bias=use_bias)
        self.fc3 = nn.Linear(fc2_units, fc3_units, bias=use_bias)
        self.fc4 = nn.Linear(fc3_units, fc4_units, bias=use_bias)
        self.fc5 = nn.Linear(fc4_units, fc5_units, bias=use_bias)
        self.fc6 = nn.Linear(fc5_units, args.action_dim_n[agent_id], bias=use_bias)
        self.reset_parameters()

    def forward(self, observation):
        """ map a state to action values
            :return: action values
        """

        if self.use_batch_norm:
            x = F.relu(self.fc1(self.bn1(observation)))
            x = F.relu(self.fc2(self.bn2(x)))
            x = F.relu(self.fc3(self.bn3(x)))
            x = F.relu(self.fc4(self.bn4(x)))
            x = F.relu(self.fc5(self.bn5(x)))
            return torch.tanh(self.fc6(self.bn6(x)))
        else:
            x = F.relu(self.fc1(observation))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = F.relu(self.fc5(x))
            return torch.tanh(self.fc6(x))

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        self.fc5.weight.data.uniform_(*hidden_init(self.fc5))
        self.fc6.weight.data.uniform_(-3e-3, 3e-3)


class Critic_v2(nn.Module):
    def __init__(self, args, agent_id, use_batch_norm=False,
                 fc1_units=128, fc2_units=64, fc3_units=32):
        """ args.hidden_dim
        :param observation_size: Dimension of each state
        :param action_size: Dimension of each state
        :param seed: random seed
        :param fc1_units: number of nodes in 1st hidden layer
        :param fc2_units: number of nodes in 2nd hidden layer
        :param fc3_units: number of nodes in 3rd hidden layer
        """
        super(Critic_v2, self).__init__()

        if args.seed is not None:
            torch.manual_seed(args.seed)

        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(args.obs_dim_n[agent_id] + args.action_dim_n[agent_id])
            self.bn2 = nn.BatchNorm1d(fc1_units)
            self.bn3 = nn.BatchNorm1d(fc2_units)
            self.bn4 = nn.BatchNorm1d(fc3_units)

        # batch norm has bias included, disable linear layer bias
        use_bias = not use_batch_norm

        self.use_batch_norm = use_batch_norm
        self.fc1 = nn.Linear(sum(args.obs_dim_n) + sum(args.action_dim_n), fc1_units)
        # self.fc1 = nn.Linear(args.obs_dim_n[agent_id] + args.action_dim_n[agent_id], fc1_units, bias=use_bias)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.reset_parameters()

    def forward(self, observation, action):
        """ map (observation, actions) pairs to Q-values
        :param observation: shape == (batch, observation_size)
        :param action: shape == (batch, action_size)
        :return: q-values values
        """
        s = torch.cat(observation, dim=1)
        a = torch.cat(action, dim=1)
        x = torch.cat([s, a], dim=1)

        #x = torch.cat([observation, action], dim=1)
        if self.use_batch_norm:
            x = F.relu(self.fc1(self.bn1(x)))
            x = F.relu(self.fc2(self.bn2(x)))
            x = F.relu(self.fc3(self.bn3(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))

        x = self.fc4(x)
        return x

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

class Critic_MATD3(nn.Module):
    def __init__(self, args):
        super(Critic_MATD3, self).__init__()
        self.fc1 = nn.Linear(sum(args.obs_dim_n) + sum(args.action_dim_n), args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

        self.fc4 = nn.Linear(sum(args.obs_dim_n) + sum(args.action_dim_n), args.hidden_dim)
        self.fc5 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc6 = nn.Linear(args.hidden_dim, 1)
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
            orthogonal_init(self.fc4)
            orthogonal_init(self.fc5)
            orthogonal_init(self.fc6)

    def forward(self, s, a):
        s = torch.cat(s, dim=1)
        a = torch.cat(a, dim=1)
        s_a = torch.cat([s, a], dim=1)

        q1 = F.relu(self.fc1(s_a))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(s_a))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2

    def Q1(self, s, a):
        s = torch.cat(s, dim=1)
        a = torch.cat(a, dim=1)
        s_a = torch.cat([s, a], dim=1)
        q1 = F.relu(self.fc1(s_a))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        return q1

# Recurrent Actor and Critic
class Actor_v3(nn.Module):
    def __init__(self, args, agent_id,layer_num = 3, hidden_layer_size: int = 128,
        max_action: float = 1.0, device = "cpu",
        unbounded: bool = False, conditioned_sigma: bool = False):
        super().__init__()
        max_action = 1.0
        self.device = device
        self.nn = nn.LSTM(
            input_size=args.obs_dim_n[agent_id], # state_shape: Sequence[int]
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            batch_first=True,
        )
        output_dim = args.action_dim_n[agent_id] # acton_shape Sequence[int]
        self.mu = nn.Linear(hidden_layer_size, output_dim)
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = nn.Linear(hidden_layer_size, output_dim)
        else:
            self.sigma_param = nn.Parameter(torch.zeros(output_dim, 1))
        self.max_action = max_action
        self._unbounded = unbounded

    def forward(self, obs, state = None):
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(-2)
        self.nn.flatten_parameters()
        if state is None:
            obs, (hidden, cell) = self.nn(obs)
        else:
            # we store the stack data in [bsz, len, ...] format
            # but pytorch rnn needs [len, bsz, ...]
            obs, (hidden, cell) = self.nn(
                obs, (
                    state["hidden"].transpose(0, 1).contiguous(),
                    state["cell"].transpose(0, 1).contiguous()
                )
            )
        logits = obs[:, -1]
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self.max_action * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), min=SIGMA_MIN, max=SIGMA_MAX).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        # First dim is batch size: [bsz, len, ...]
        return mu

class Critic_v3(nn.Module):
    def __init__(self, args, agent_id, layer_num = 3, device="cpu", hidden_layer_size: int = 512):
        super().__init__()
        self.device = device
        self.nn = nn.LSTM(
            input_size=1, # args.obs_dim_n[agent_id],
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            batch_first=True,
        )
        # The first dimension of the linear layer is the sum of the action dim and hidden layer
        # WORKS self.lin_layer = sum(args.action_dim_n) + hidden_layer_size
        self.fc2 = nn.Linear(hidden_layer_size, 1)

    def forward(self, obs, act):
        self.nn.flatten_parameters()
        # Concatenate the action and observation
        obs = torch.cat(obs, dim=1)
        act = torch.cat(act, dim=1)
        o_a = torch.cat([obs, act], dim=1)

        # Add dimensions for the LSTM, process with the LSTM, reshape and return
        o_a = o_a.unsqueeze(2)
        obs, (hidden, cell) = self.nn(o_a)
        obs = obs[:, -1]
        # o_a_2 = torch.cat([obs, act], dim=1)
        obs = self.fc2(obs)
        return obs
