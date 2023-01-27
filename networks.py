
import torch.nn as nn
from torch.distributions import Categorical
import torch
class ActorCriticBackbone(nn.Module):
    """This is the backbone for feature extraction of both the Actor and The critic networks. They share the same
    hidden layers but differ in the final head.
    """
    def __init__(self):
        super(ActorCriticBackbone, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=6, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels= 128, kernel_size=3, stride=2)
        self.fc = nn.Linear(128*9*9,  out_features=512)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 128 * 9 * 9)
        x = torch.relu(self.fc(x))
        return x

class Actor(nn.Module):
    def __init__(self, num_actions):
        super(Actor, self).__init__()
        self.fc = nn.Linear(512, num_actions)

    def forward(self, x):
        x = torch.softmax(self.fc(x), dim=-1)
        return x

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.fc(x)
        return x

class ActorCritic(nn.Module):
    def __init__(self, num_actions):
        super(ActorCritic, self).__init__()
        self.Backbone = ActorCriticBackbone()
        self.actor = Actor(num_actions)
        self.critic = Critic()

    def forward(self, x):
        x = self.Backbone(x)
        probs = self.actor(x)
        dist = Categorical(probs)
        value = self.critic(x)
        return dist, value
    