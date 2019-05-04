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

        # Value Net
        self.value_layer1 = nn.Linear(layer_widths[3] * self.input_layer_width, layer_widths[4])
        self.vbn = nn.BatchNorm1d(layer_widths[4])
        self.value_layer2 = nn.Linear(layer_widths[4], 1)

        # Advantage Net
        self.advantage_layer1 = nn.Linear(layer_widths[3] * self.input_layer_width, layer_widths[4])
        self.abn = nn.BatchNorm1d(layer_widths[4])
        self.advantage_layer2 = nn.Linear(layer_widths[4], self.num_outputs)

        # Final frame prediction
        self.state_prediction_layer = nn.Linear(layer_widths[3] * self.input_layer_width, self.input_layer_width)

    def forward(self, x):
        x1 = F.relu(self.bn0(self.conv0(x)))
        x2 = F.relu(self.bn1(self.conv1(x1 + x)))
        x3 = F.relu(self.bn2(self.conv2(x2 + x1)))
        conv_x = x3.view(x3.size(0), -1)

        # Q function layers:
        value = F.relu(self.vbn(self.value_layer1(conv_x)))
        value = self.value_layer2(value)

        advg = F.relu(self.abn(self.advantage_layer1(conv_x)))
        advg = self.advantage_layer2(advg)

        q_val = value.expand(-1, self.num_outputs) + (advg - advg.mean(1, keepdim=True))

        # State layers
        # Tanh + etc. to turn this into a probability
        state_prediction = (F.tanh(self.state_prediction_layer(conv_x).view(-1, 1, self.h, self.w) + 2 * x) + 1) / 2

        return q_val, state_prediction
