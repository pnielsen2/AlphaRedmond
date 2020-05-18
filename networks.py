import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.blocks = 9
        self.conv_block_conv = nn.Conv2d(18, 64, 3, padding = 1)
        self.conv_block_batch_norm = nn.BatchNorm2d(64)

        self.resid_block_conv_1s = [nn.Conv2d(64, 64, 3, padding = 1) for i in range(self.blocks)]
        self.resid_block_batch_norm_1s = [nn.BatchNorm2d(64) for i in range(self.blocks)]

        self.resid_block_conv_2s = [nn.Conv2d(64, 64, 3, padding = 1) for i in range(self.blocks)]
        self.resid_block_batch_norm_2s = [nn.BatchNorm2d(64) for i in range(self.blocks)]

        self.policy_conv = nn.Conv2d(64, 2, 1)
        self.policy_batch_norm = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(162, 82)

        self.value_conv = nn.Conv2d(64, 1, 1)
        self.value_batch_norm = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(81, 64)
        self.value_fc2 = nn.Linear(64, 1)



    def resid_tower(self, input):
        for i in range(self.blocks):
            intermediate_activation = F.relu(self.resid_block_batch_norm_1s[i](self.resid_block_conv_1s[i](input)))
            block_output = F.relu(self.resid_block_batch_norm_2s[i](self.resid_block_conv_2s[i](intermediate_activation)) + input)
            input = block_output
        return block_output

    def forward(self, x):
        convolutional_block = F.relu(self.conv_block_batch_norm(self.conv_block_conv(x)))
        residual_tower = self.resid_tower(convolutional_block)
        policy = self.policy_fc(F.relu(self.policy_batch_norm(self.policy_conv(residual_tower))).view(-1))
        value = torch.tanh(self.value_fc2(F.relu(self.value_fc1(F.relu(self.value_batch_norm(self.value_conv(residual_tower)).view(-1))))))

        return policy, value

class FastNetwork(nn.Module):
    def __init__(self):
        super(FastNetwork, self).__init__()
        self.blocks = 9
        self.conv_block_conv = nn.Conv2d(18, 64, 3, padding = 1)
        self.conv_block_batch_norm = nn.BatchNorm2d(64)

        self.resid_block_conv_1s = [nn.Conv2d(64, 64, 3, padding = 1) for i in range(self.blocks)]
        self.resid_block_batch_norm_1s = [nn.BatchNorm2d(64) for i in range(self.blocks)]

        self.resid_block_conv_2s = [nn.Conv2d(64, 64, 3, padding = 1) for i in range(self.blocks)]
        self.resid_block_batch_norm_2s = [nn.BatchNorm2d(64) for i in range(self.blocks)]

        self.policy_conv = nn.Conv2d(64, 2, 1)
        self.policy_batch_norm = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(162, 82)

        self.value_conv = nn.Conv2d(64, 1, 1)
        self.value_batch_norm = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(81, 64)
        self.value_fc2 = nn.Linear(64, 1)



    def resid_tower(self, input):
        for i in range(self.blocks):
            intermediate_activation = F.relu(self.resid_block_batch_norm_1s[i](self.resid_block_conv_1s[i](input)))
            block_output = F.relu(self.resid_block_batch_norm_2s[i](self.resid_block_conv_2s[i](intermediate_activation)) + input)
            input = block_output
        return block_output

    def forward(self, x):
        policy = torch.randn(82)
        value = torch.randn(1)

        return policy, value
