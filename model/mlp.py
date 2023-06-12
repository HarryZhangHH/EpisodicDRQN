import torch
import torch.nn as nn
from utils import Type

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # hidden layer
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: Type.TensorType):
        x = x.type(torch.FloatTensor).to(device)
        x = x.view(-1, self.input_size)
        logits = self.linear_relu_stack(x)
        return logits