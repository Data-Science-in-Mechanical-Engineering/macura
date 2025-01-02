import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2
    #Reset Parameters towards network_reset_factor randomness
    def reset_weights(self, network_reset_factor):
        for module in self.children():
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    random_weights = torch.zeros_like(module.weight)
                    torch.nn.init.xavier_uniform_(random_weights, gain=1)
                    module.weight = torch.nn.Parameter((1-network_reset_factor)*module.weight + network_reset_factor*random_weights)
                    module.bias = torch.nn.Parameter((1-network_reset_factor)*module.bias)

#Q Network with Layer normalization
class QNetworkLN(QNetwork):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetworkLN, self).__init__(num_inputs, num_actions, hidden_dim)
        self.hidden_dim=hidden_dim


    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(F.layer_norm(self.linear1(xu), [self.hidden_dim]))
        x1 = F.relu(F.layer_norm(self.linear2(x1), [self.hidden_dim]))
        x1 = self.linear3(x1)

        x2 = F.relu(F.layer_norm(self.linear4(xu), [self.hidden_dim]))
        x2 = F.relu(F.layer_norm(self.linear5(x2), [self.hidden_dim]))
        x2 = self.linear6(x2)

        return x1, x2

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            high = np.array(action_space.high)
            low = np.array(action_space.low)
            self.action_scale = torch.FloatTensor((high - low) / 2.0)
            self.action_bias = torch.FloatTensor((high + low) / 2.0)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def sample_using_eps(self, state, eps):
        mean, log_std = self.forward(state)
        # print(mean.shape, log_std.shape, eps.shape)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = mean + std * eps  # normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

class GaussianPolicyLN(GaussianPolicy):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicyLN, self).__init__(num_inputs, num_actions, hidden_dim, action_space)
        self.hidden_dim=hidden_dim
    def forward(self, state):
        x = F.relu(F.layer_norm(self.linear1(state),[self.hidden_dim]))
        x = F.relu(F.layer_norm(self.linear2(x),[self.hidden_dim]))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.0
            self.action_bias = 0.0
        else:
            high = np.array(action_space.high)
            low = np.array(action_space.low)
            self.action_scale = torch.FloatTensor((high - low) / 2.0)
            self.action_bias = torch.FloatTensor((high + low) / 2.0)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0.0, std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.0), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)

class DeterministicPolicyLN(DeterministicPolicy):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicyLN, self).__init__(num_inputs, num_actions, hidden_dim, action_space)
        self.hidden_dim=hidden_dim
    def forward(self, state):
        x = F.relu(F.layer_norm(self.linear1(state),[self.hidden_dim]))
        x = F.relu(F.layer_norm(self.linear2(x),[self.hidden_dim]))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean