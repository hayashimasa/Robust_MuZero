"""Network architectures for MuZero

Author: Masahiro Hayashi

This script defines the network architectures for MuZero, which consists of
3 components: Representation, Dynamic, and Prediction. The main MuZero network
can perform recurrent rollouts as described in the orginal paper.

    https://arxiv.org/pdf/1911.08265.pdf

This implementation is modified from the following MuZero implementation:

    https://github.com/werner-duvaud/muzero-general

The file is organized into 3 sections:
    - Conversion tools
    - Building blocks for defining networks
    - Components of MuZero
    - Main architecture of MuZero
"""
from abc import ABC, abstractmethod
import math

import torch
from torch import nn

###############################################################################
# Tools
###############################################################################
def support_to_scalar(logits, support_size):
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    # Decode to a scalar
    probabilities = torch.softmax(logits, dim=1)
    support = (
        torch.tensor([x for x in range(-support_size, support_size + 1)])
        .expand(probabilities.shape)
        .float()
        .to(device=probabilities.device)
    )
    x = torch.sum(support * probabilities, dim=1, keepdim=True)
    eps = 0.001
    # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (
        ((torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps)) - 1) / (2 * eps))
        ** 2
        - 1
    )
    return x

def scalar_to_support(x, support_size):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    eps = 0.001
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x

    # Encode on a vector
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
    logits.scatter_(
        2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
    )
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits

def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict

###############################################################################
# Building Blocks
###############################################################################
class ConvBnReLU(nn.Module):
    """Convoutional-Batch Normalization-ReLU block
    """
    def __init__(
        self, in_dim, out_dim,
        filter_size=3, stride=1, padding=1, useReLU=True
    ):
        self.name = 'Conv-BN-ReLU'
        super(ConvBnReLU, self).__init__()
        self.useReLU = useReLU
        self.conv = nn.Conv2d(in_dim, out_dim, filter_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_dim)
        if self.useReLU:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.useReLU:
            x = self.relu(x)
        return x

class ConvBnLeakyReLU(ConvBnReLU):
    """Convoutional-Batch Normalization-LeakyReLU block
    """
    def __init__(self, in_dim, out_dim, filter_size=3, stride=1, padding=1):
        super(ConvBnLeakyReLU, self).__init__(in_dim, out_dim)
        self.name = 'Conv-BN-LeakyReLU'
        self.relu = nn.LeakyReLU(2e-1, inplace=True)

class ResidualBlock(nn.Module):
    """Basic Residiual Block
    """
    def __init__(
        self, in_dim, out_dim, padding=1, padding_type='zero',
        downsample=None, stride=1, use_dropout=False
    ):
        self.name = 'Basic Residual Block'
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        # padding options
        _pad = {
            'reflect': nn.ReflectionPad2d(padding),
            'replicate': nn.ReplicationPad2d(padding)
        }
        self.block = []
        if padding_type == 'zero':
            self.padding = 1
        else:
            self.padding = 0
            self.pad = _pad.get(padding_type, None)
            if self.pad is None:
                NotImplementedError(
                    f'padding [{padding_type}] is not implemented'
                )
        # conv-bn-relu 1
        if self.padding == 0:
            self.block.append(self.pad)
        conv1 = ConvBnReLU(in_dim, out_dim, 3, stride, padding)
        self.block.append(conv1)
        # dropout
        if use_dropout:
            dropout = nn.Dropout2d(0.5) if use_dropout else None
            self.block.append(dropout)
        # conv-bn-relu 2
        if self.padding == 0:
            self.block.append(self.pad)
        conv2 = ConvBnReLU(out_dim, out_dim, 3, stride, padding, useReLU=False)
        self.block.append(conv2)
        # initialize block
        self.block = nn.Sequential(*self.block)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        identity = x
        # if not self.downsample is None:
        #     identity = self.downsample(identity)
        residual = self.block(x)
        out = identity + residual
        out = self.relu(out)
        return out

class ResidualBlocks(nn.Module):
    """Multiple residual blocks
    """
    def __init__(
        self, n_blocks, in_dim, out_dim, padding=1, padding_type='zero',
        downsample=None, stride=1, use_dropout=False
    ):
        self.name = "Residual Blocks"
        super(ResidualBlocks, self).__init__()
        self.blocks = [
            ResidualBlock(
                in_dim, out_dim, padding, padding_type,
                downsample, stride, use_dropout
            ) for _ in range(n_blocks)
        ]
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        return self.blocks(x)

class MLP(nn.Module):
    """Multi-layer Perceptron
    """
    def __init__(
        self, in_dim, out_dim, h_dims=[],
        activation=torch.nn.ReLU, out_activation=torch.nn.Tanh
    ):
        self.name = "Multi-layer Perceptron"
        super(MLP, self).__init__()
        dims = [in_dim] + h_dims + [out_dim]
        self.layers = []
        for i in range(len(dims)-1):
            self.layers += [nn.Linear(dims[i], dims[i+1]), activation(True)]
        self.layers[-1] = out_activation()
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)

class AbstractNetwork(ABC, nn.Module):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def initial_inference(self, observation):
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state, action):
        pass

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)

###############################################################################
# MuZero Components
###############################################################################
class Downsample(nn.Module):
    """Downsample Module
    """
    def __init__(self, in_dim, out_dim):
        self.name = "Downsample Module"
        super(Downsample, self).__init__()
        self.conv1 = ConvBnReLU(in_dim, out_dim // 2, stride=2)
        self.resblocks1 = ResidualBlocks(2, out_dim // 2, out_dim // 2)
        self.conv2 = ConvBnReLU(out_dim // 2, out_dim, stride=2)
        self.resblocks2 = ResidualBlocks(3, out_dim, out_dim)
        self.avg_pool1 = nn.AvgPool2d(3, stride=2, padding=1)
        self.resblocks3 = ResidualBlocks(3, out_dim, out_dim)
        self.avg_pool2 = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.resblocks1(x)
        x = self.conv2(x)
        x = self.resblocks2(x)
        x = self.avg_pool1(x)
        x = self.resblocks3(x)
        x = self.avg_pool2(x)
        return x

class RepresentationNet(nn.Module):
    """Representation Network

    Encodes past obervations into a state representation
    """
    def __init__(
        self, observation_shape, n_observations,
        n_blocks=16, out_dim=256, downsample=True
    ):
        self.name = 'Representation Network'
        super(RepresentationNet, self).__init__()
        # in_dim = (observation_shape[0] + 1) * n_observations + 3
        in_dim = observation_shape[0] * (1 + n_observations) + n_observations
        self.use_downsample = downsample
        if self.use_downsample:
            self.downsample = Downsample(in_dim, out_dim)
        else:
            self.convblock = ConvBnReLU(in_dim, out_dim)
        self.resblocks = ResidualBlocks(n_blocks, out_dim, out_dim)

    def forward(self, x):
        if self.use_downsample:
            x = self.downsample(x)
        else:
            x = self.convblock(x)
        out = self.resblocks(x)
        return out

class DynamicsNet(nn.Module):
    """Dynamics Network

    Outputs an immediate reward and an internal state
    """
    def __init__(
        self, n_blocks=16, dim=256,
        reward_dim=256, fc_reward_h_dim=256,
        support_dim=2*300+1, conv_dim=256*6*6,
    ):
        self.name = 'Dynamics Network'
        super(DynamicsNet, self).__init__()
        # input has an additional action plane
        self.convblock = ConvBnReLU(dim+1, dim)
        self.resblocks = ResidualBlocks(n_blocks, dim, dim)
        # reward
        self.conv_r = torch.nn.Conv2d(dim, reward_dim, 1)
        self.conv_dim = conv_dim
        self.fc = MLP(conv_dim, support_dim, fc_reward_h_dim)

    def forward(self, s_old):
        s_old = self.convblock(s_old)
        s_old = self.resblocks(s_old)
        s_new = s_old
        r = self.conv_r(s_old)
        r = r.view(-1, self.conv_dim)
        r = self.fc(r)
        return r, s_new

class PredictionNet(nn.Module):
    """Prediction Network

    Outputs an a policy and a value
    """
    def __init__(
        self,
        n_action=4,
        n_blocks=16,
        dim=256,
        value_dim=256,
        policy_dim=256,
        fc_value_h_dim=None,
        fc_policy_h_dim=None,
        support_dim=2*300*1,
        conv_value_dim=256*6*6,
        conv_policy_dim=256*6*6
    ):
        self.name = 'Prediction Network'
        super(PredictionNet, self).__init__()
        self.resblocks = ResidualBlocks(n_blocks, dim, dim)
        # policy and value
        self.conv_p = torch.nn.Conv2d(dim, policy_dim, 1)
        self.conv_v = torch.nn.Conv2d(dim, value_dim, 1)
        self.conv_p_dim = conv_policy_dim
        self.conv_v_dim = conv_value_dim
        self.fc_p = MLP(conv_policy_dim, n_action, fc_policy_h_dim)
        self.fc_v = MLP(conv_value_dim, support_dim, fc_value_h_dim)

    def forward(self, s):
        s = self.resblocks(s)
        p = self.conv_p(s)
        v = self.conv_v(s)
        p = p.view(-1, self.conv_p_dim)
        v = v.view(-1, self.conv_v_dim)
        p = self.fc_p(p)
        v = self.fc_v(v)
        return p, v

###############################################################################
# Main MuZero Architecture
###############################################################################
class MuZeroResidualNetwork(AbstractNetwork):
    """MuZero Network
    """
    def __init__(
        self,
        observation_shape,
        n_observations,
        n_actions,
        n_blocks,
        dim,
        reward_dim,
        value_dim,
        policy_dim,
        fc_reward_h_dim,
        fc_value_h_dim,
        fc_policy_h_dim,
        support_size,
        downsample,
    ):
        super().__init__()
        self.action_space_size = n_actions
        self.full_support_size = 2 * support_size + 1
        self.observation_shape = observation_shape
        self.downsample = downsample
        conv_r_dim = self.get_conv_dim(reward_dim)
        conv_p_dim = self.get_conv_dim(policy_dim)
        conv_v_dim = self.get_conv_dim(value_dim)

        self.representation_network = torch.nn.DataParallel(
            RepresentationNet(
                observation_shape,
                n_observations,
                n_blocks,
                dim,
                downsample,
            )
        )

        self.dynamics_network = torch.nn.DataParallel(
            DynamicsNet(
                n_blocks,
                dim,
                reward_dim,
                fc_reward_h_dim,
                self.full_support_size,
                conv_r_dim,
            )
        )

        self.prediction_network = torch.nn.DataParallel(
            PredictionNet(
                n_actions,
                n_blocks,
                dim,
                value_dim,
                policy_dim,
                fc_value_h_dim,
                fc_policy_h_dim,
                self.full_support_size,
                conv_v_dim,
                conv_p_dim,
            )
        )

    def get_conv_dim(self, dim):
        """Calculate the output dimension of a convolutional block
        """
        if self.downsample:
            dim *= math.ceil(self.observation_shape[1] / 16)
            dim *= math.ceil(self.observation_shape[2] / 16)
        else:
            dim *= self.observation_shape[1] * self.observation_shape[2]
        return dim

    def get_bound(self, s, bound):
        s = s.view(-1, s.shape[1], s.shape[2] * s.shape[3])
        s = bound(s, 2, keepdim=True)[0]
        s = s.unsqueeze(-1)
        return s

    def scale(self, state):
        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_state = self.get_bound(state, torch.min)
        max_state = self.get_bound(state, torch.max)
        scale_state = max_state - min_state
        scale_state[scale_state < 1e-5] += 1e-5
        state_normalized = (state - min_state) / scale_state
        return state_normalized

    def prediction(self, encoded_state):
        policy, value = self.prediction_network(encoded_state)
        return policy, value

    def representation(self, observation):
        encoded_state = self.representation_network(observation)
        encoded_state_normalized = self.scale(encoded_state)
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action
        # (See paper appendix Network Architecture)
        s_shape = encoded_state.shape
        action_one_hot = torch.ones((s_shape[0], 1, s_shape[2], s_shape[3]))
        action_one_hot = action_one_hot.to(action.device).float()
        action_one_hot = action[:, :, None, None] * action_one_hot
        action_one_hot /= self.action_space_size
        s_a = torch.cat((encoded_state, action_one_hot), dim=1)
        reward, next_encoded_state = self.dynamics_network(s_a)
        next_encoded_state_normalized = self.scale(next_encoded_state)
        return reward, next_encoded_state_normalized

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        # reward equal to 0 for consistency
        reward = torch.zeros(1, self.full_support_size)
        support_size = self.full_support_size // 2
        reward = reward.scatter(1, torch.tensor([[support_size]]).long(), 1.0)
        reward = reward.repeat(len(observation), 1).to(observation.device)
        reward = torch.log(reward)
        return value, reward, policy_logits, encoded_state

    def recurrent_inference(self, encoded_state, action):
        reward, next_encoded_state = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state

class MuZeroNetwork:
    def __new__(cls, config):
        # if config.network == "fullyconnected":
        #     return MuZeroFullyConnectedNetwork(
        #         config.observation_shape,
        #         config.stacked_observations,
        #         len(config.action_space),
        #         config.encoding_size,
        #         config.fc_reward_layers,
        #         config.fc_value_layers,
        #         config.fc_policy_layers,
        #         config.fc_representation_layers,
        #         config.fc_dynamics_layers,
        #         config.support_size,
        #     )
        if config.network == "resnet":
            return MuZeroResidualNetwork(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.blocks,
                config.channels,
                config.reduced_channels_reward,
                config.reduced_channels_value,
                config.reduced_channels_policy,
                config.resnet_fc_reward_layers,
                config.resnet_fc_value_layers,
                config.resnet_fc_policy_layers,
                config.support_size,
                config.downsample,
            )
        else:
            raise NotImplementedError(
                'The network parameter should be "fullyconnected" or "resnet".'
            )

###############################################################################
# For testing
###############################################################################
if __name__ == '__main__':
    # # Deepmind Atari hyperparameters
    # observation_shape = (3, 96, 96)
    # n_observations = 32
    # n_blocks = 16
    # n_actions = 4
    # dim = 256
    # reward_dim = 256
    # value_dim = 256
    # policy_dim = 256
    # fc_reward_h_dim = [256, 256]
    # fc_policy_h_dim = [256, 256]
    # fc_value_h_dim = [256, 256]
    # support_size = 300
    # downsample = 'resnet'

    # light Atari hyperparameters
    observation_shape = (3, 96, 96)
    n_observations = 0
    n_blocks = 2
    n_actions = 4
    dim = 16
    reward_dim = 4
    value_dim = 4
    policy_dim = 4
    fc_reward_h_dim = [16]
    fc_policy_h_dim = [16]
    fc_value_h_dim = [16]
    support_size = 10
    downsample = 'resnet'
    # MuZero main network testing
    MuZeroNet = MuZeroResidualNetwork(
        observation_shape, n_observations, n_actions, n_blocks, dim,
        reward_dim, value_dim, policy_dim,
        fc_reward_h_dim, fc_value_h_dim, fc_policy_h_dim,
        support_size, downsample
    )
    K = 5
    # O = torch.rand((1, 128, 96, 96))
    in_dim = (observation_shape[0]+1) * n_observations + observation_shape[0]
    O = torch.rand((1, in_dim, 96, 96))
    v, r, p, s = MuZeroNet.initial_inference(O)
    print('value', v.shape)
    print('reward', r.shape)
    print('policy', p.shape)
    print('state', s.shape)
    s_k = s
    a = torch.ones((1, 1)) * torch.argmax(p)
    for _ in range(K-1):
        v_k, r_k, p_k, s_k = MuZeroNet.recurrent_inference(s_k, a)
        a = torch.ones((1, 1)) * torch.argmax(p)
    print('value_k', v_k.shape)
    print('reward_k', r_k.shape)
    print('policy_k', p_k.shape)
    print('state_k', s_k.shape)
    # s = torch.rand((1,17, 6, 6))
    # dyna_net = DynamicsNet(n_blocks, dim, reward_dim, fc_reward_h_dim,
    #     support_size*2+1, 4*6*6)
    # r, s = dyna_net(s)
    # print(r.size())
    # print(s.size())
