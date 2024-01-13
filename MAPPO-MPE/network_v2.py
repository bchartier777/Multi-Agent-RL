import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import *
import numpy as np

SIGMA_MIN = -20
SIGMA_MAX = 2

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)

class Actor_RNN(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(actor_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2, gain=0.01)

    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size*N, actor_input_dim),prob.shape=(mini_batch_size*N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        prob = torch.softmax(self.fc2(self.rnn_hidden), dim=-1)
        return prob


class Critic_RNN(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(critic_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size*N, critic_input_dim), value.shape=(mini_batch_size*N, 1)
        x = self.activate_func(self.fc1(critic_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        value = self.fc2(self.rnn_hidden)
        return value


class Actor_RNN_v2(nn.Module):
    def __init__(self, args, actor_input_dim, layer_num = 3, hidden_layer_size: int = 64,
                 device = "cpu", unbounded: bool = False, conditioned_sigma: bool = False):
        super().__init__()
        self.device = device
        self.nn = nn.LSTM(
            input_size=actor_input_dim, # state_shape: Sequence[int]
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            batch_first=True,
        )
        output_dim = args.action_dim # was args.state_dim  acton_shape Sequence[int]
        self.mu = nn.Linear(hidden_layer_size, output_dim)
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = nn.Linear(hidden_layer_size, output_dim)
        else:
            self.sigma_param = nn.Parameter(torch.zeros(output_dim, 1))
        self.max_action = args.max_action
        self._unbounded = unbounded

    def forward(self, actor_input, state = None):
        if len( actor_input.shape) == 2:
             actor_input =  actor_input.unsqueeze(-2)
        self.nn.flatten_parameters()
        if state is None:
            act_input_nn, (hidden, cell) = self.nn(actor_input)
        else:
            # we store the stack data in [bsz, len, ...] format
            # but pytorch rnn needs [len, bsz, ...]
            act_input_nn, (hidden, cell) = self.nn(
                act_input_nn, (
                    state["hidden"].transpose(0, 1).contiguous(),
                    state["cell"].transpose(0, 1).contiguous()
                )
            )
        logits = act_input_nn[:, -1]
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
        mu_sm = torch.softmax(mu, dim=-1)
        return mu_sm

class Critic_RNN_v2(nn.Module):
    # def __init__(self, args, critic_input_dim):
    def __init__(self, args, critic_input_dim, layer_num = 3, hidden_layer_size: int = 64,
                 device = "cpu", unbounded: bool = False, conditioned_sigma: bool = False):
        super().__init__()
        self.device = device
        self.nn = nn.LSTM(
            input_size=critic_input_dim, # state_shape: Sequence[int]
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            batch_first=True,
        )
        output_dim = 1 # See GRU Critic dims
        self.mu = nn.Linear(hidden_layer_size, output_dim)
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = nn.Linear(hidden_layer_size, output_dim)
        else:
            self.sigma_param = nn.Parameter(torch.zeros(output_dim, 1))
        self.max_action = args.max_action
        self._unbounded = unbounded

    def forward(self, critic_input, state = None):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size*N, critic_input_dim), value.shape=(mini_batch_size*N, 1)
        if len( critic_input.shape) == 2:
             critic_input =  critic_input.unsqueeze(-2)
        self.nn.flatten_parameters()
        if state is None:
            act_input_nn, (hidden, cell) = self.nn(critic_input)
        else:
            # we store the stack data in [bsz, len, ...] format
            # but pytorch rnn needs [len, bsz, ...]
            act_input_nn, (hidden, cell) = self.nn(
                act_input_nn, (
                    state["hidden"].transpose(0, 1).contiguous(),
                    state["cell"].transpose(0, 1).contiguous()
                )
            )
        logits = act_input_nn[:, -1]
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
        mu_sm = torch.softmax(mu, dim=-1)
        return mu_sm


class Actor_MLP(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_MLP, self).__init__()
        self.fc1 = nn.Linear(actor_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size, episode_limit, N, actor_input_dim), prob.shape(mini_batch_size, episode_limit, N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        x = self.activate_func(self.fc2(x))
        prob = torch.softmax(self.fc3(x), dim=-1)
        return prob


class Critic_MLP(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_MLP, self).__init__()
        self.fc1 = nn.Linear(critic_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size, episode_limit, N, critic_input_dim), value.shape=(mini_batch_size, episode_limit, N, 1)
        x = self.activate_func(self.fc1(critic_input))
        x = self.activate_func(self.fc2(x))
        value = self.fc3(x)
        return value

class Actor_MLP_v2(nn.Module):
    # Need: args.obs_dim_n (actor_input_dim), args.action_dim_n (args.action_dim)
    # def __init__(self, args, actor_input_dim):
    def __init__(self, args, actor_input_dim, use_batch_norm=False,
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
        super(Actor_MLP_v2, self).__init__()

        if args.seed is not None:
            torch.manual_seed(args.seed)

        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(actor_input_dim)
            self.bn2 = nn.BatchNorm1d(fc1_units)
            self.bn3 = nn.BatchNorm1d(fc2_units)
            self.bn4 = nn.BatchNorm1d(fc3_units)
            self.bn5 = nn.BatchNorm1d(fc4_units)
            self.bn6 = nn.BatchNorm1d(fc5_units)

        # batch norm has bias included, disable linear layer bias
        use_bias = not use_batch_norm

        self.use_batch_norm = use_batch_norm
        self.fc1 = nn.Linear(actor_input_dim, fc1_units, bias=use_bias)
        self.fc2 = nn.Linear(fc1_units, fc2_units, bias=use_bias)
        self.fc3 = nn.Linear(fc2_units, fc3_units, bias=use_bias)
        self.fc4 = nn.Linear(fc3_units, fc4_units, bias=use_bias)
        self.fc5 = nn.Linear(fc4_units, fc5_units, bias=use_bias)
        self.fc6 = nn.Linear(fc5_units, args.action_dim, bias=use_bias)
        self.reset_parameters()

    def forward(self, actor_input):
        """ 
        """
        if self.use_batch_norm:
            x = F.relu(self.fc1(self.bn1(actor_input)))
            x = F.relu(self.fc2(self.bn2(x)))
            x = F.relu(self.fc3(self.bn3(x)))
            x = F.relu(self.fc4(self.bn4(x)))
            x = F.relu(self.fc5(self.bn5(x)))
            return torch.tanh(self.fc6(self.bn6(x)))
        else:
            x = F.relu(self.fc1(actor_input))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            prob = torch.softmax(self.fc5(x), dim=-1)
            return prob

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        self.fc5.weight.data.uniform_(*hidden_init(self.fc5))
        self.fc6.weight.data.uniform_(-3e-3, 3e-3)

class Critic_MLP_v2(nn.Module):
    def __init__(self, args, critic_input_dim, use_batch_norm=False,
                 fc1_units=128, fc2_units=64, fc3_units=32):
        """ args.hidden_dim
        :param observation_size: Dimension of each state
        :param action_size: Dimension of each state
        :param seed: random seed
        :param fc1_units: number of nodes in 1st hidden layer
        :param fc2_units: number of nodes in 2nd hidden layer
        :param fc3_units: number of nodes in 3rd hidden layer
        """
        super(Critic_MLP_v2, self).__init__()

        if args.seed is not None:
            torch.manual_seed(args.seed)

        # Batch norm requires action space 
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(critic_input_dim)
            self.bn2 = nn.BatchNorm1d(fc1_units)
            self.bn3 = nn.BatchNorm1d(fc2_units)
            self.bn4 = nn.BatchNorm1d(fc3_units)

        # batch norm has bias included, disable linear layer bias
        use_bias = not use_batch_norm

        self.use_batch_norm = use_batch_norm
        self.fc1 = nn.Linear(critic_input_dim, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.reset_parameters()

    def forward(self, critic_input):
        if self.use_batch_norm:
            input = torch.squeeze(critic_input, dim=3)
            x = F.relu(self.fc1(self.bn1(input)))
            x = F.relu(self.fc2(self.bn2(x)))
            x = F.relu(self.fc3(self.bn3(x)))
        else:
            x = F.relu(self.fc1(critic_input))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))

        x = self.fc4(x)
        return x

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)
