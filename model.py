import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise
    """

    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
        if bias:
            self.sigma_bias = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))

    def forward(self, input):
        torch.randn(self.epsilon_input.size(), out=self.epsilon_input)
        torch.randn(self.epsilon_output.size(), out=self.epsilon_output)

        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        eps_in = func(self.epsilon_input)
        eps_out = func(self.epsilon_output)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * Variable(eps_out.t())
        noise_v = Variable(torch.mul(eps_in, eps_out))
        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)


class DQN(nn.Module):
    def __init__(self, h, w, num_outputs):
        super().__init__()
        self.h = h
        self.w = w
        self.input_layer_width = h * w
        self.num_outputs = num_outputs

        # Encoder section
        layer_widths = [
            1,
            64,
            64,
            64,
            128,
        ]
        self.conv0 = nn.Conv2d(layer_widths[0], layer_widths[1], 3, padding=1)
        self.bn0 = nn.BatchNorm2d(layer_widths[1])
        self.conv1 = nn.Conv2d(layer_widths[1], layer_widths[2], 3, padding=1)
        self.bn1 = nn.BatchNorm2d(layer_widths[2])
        self.conv2 = nn.Conv2d(layer_widths[2], layer_widths[3], 3, padding=1)
        self.bn2 = nn.BatchNorm2d(layer_widths[3])

        # Split layer size
        split_layer_size = layer_widths[3] * self.input_layer_width + 16 + self.h * self.w

        # Value Net
        self.value_layer1 = nn.Linear(split_layer_size, layer_widths[4])
        self.vbn = nn.BatchNorm1d(layer_widths[4])
        self.value_layer2 = nn.Linear(layer_widths[4], 1)

        # Advantage Net
        self.advantage_layer1 = nn.Linear(split_layer_size, layer_widths[4])
        self.abn = nn.BatchNorm1d(layer_widths[4])
        self.advantage_layer2 = NoisyFactorizedLinear(layer_widths[4], self.num_outputs)

        # Final frame prediction
        self.state_prediction_layer = nn.Linear(split_layer_size, self.input_layer_width)

    def forward(self, state, piece):
        conv1 = F.relu(self.bn0(self.conv0(state)))
        conv2 = F.relu(self.bn1(self.conv1(conv1 + state)))
        conv3 = F.relu(self.bn2(self.conv2(conv2 + conv1)))
        conv_and_piece = torch.cat([
            conv3.view(conv3.size(0), -1),
            piece.view(piece.size(0), -1),
            state.view(state.size(0), -1)
        ], dim=1)

        # Advantage layers (relative advantage of state over other states)
        advg = F.relu(self.abn(self.advantage_layer1(conv_and_piece)))
        advg = self.advantage_layer2(advg)

        # # Only need to evaluate advantage layer for deciding actions
        # # Comment out for debug purposes
        # if self.training:
        #     return advg

        # Value layers (value of current state):
        value = F.relu(self.vbn(self.value_layer1(conv_and_piece)))
        value = self.value_layer2(value)

        q_val = value.expand(-1, self.num_outputs) + (advg - advg.mean(1, keepdim=True))

        # State layers
        # Sigmoid + etc. to turn this into a probability is done in the loss function
        # Bias by current state
        state_prediction = self.state_prediction_layer(conv_and_piece).view(-1, 1, self.h, self.w) + state

        return q_val, state_prediction
