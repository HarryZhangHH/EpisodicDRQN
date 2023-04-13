import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Type

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: Type.TensorType):
        # Set initial hidden and cell states
        # x need to be: (batch_size, seq_length, input_size)   seq_length=config.h
        x = x.type(torch.FloatTensor).to(device)
        x = x.view(x.size(0), -1, self.input_size)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :])
        return out


class LSTMVariant(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, feature_size: int, output_size: int,
                 hidden_size_f: int):
        super(LSTMVariant, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.feature_size = feature_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        if self.feature_size != 0:
            self.fc1 = nn.Linear(feature_size, hidden_size_f)
            self.fc1_bn = nn.BatchNorm1d(hidden_size + hidden_size_f)
            self.fc2 = nn.Linear(hidden_size + hidden_size_f, hidden_size)
            self.fc2_bn = nn.BatchNorm1d(hidden_size)
            self.dropout1 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x: Type.TensorStructType):
        x1, x2 = x[0], x[1]
        x1 = x1.type(torch.FloatTensor).to(device)
        x1 = x1.view(x1.size(0), -1, self.input_size)
        x2 = x2.type(torch.FloatTensor).to(device)
        x2 = x2.view(x2.size(0), -1)

        h0 = torch.zeros(self.num_layers, x1.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x1.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        self.lstm.flatten_parameters()
        out_lstm, _ = self.lstm(x1, (h0, c0))  # out_lstm: tensor of shape (batch_size, seq_length, hidden_size)

        if self.feature_size != 0:
            # LSTM Varient
            out_fc1 = self.fc1(x2)
            x = torch.cat((out_lstm[:, -1, :].view(x1.size(0), self.hidden_size), out_fc1.view(x1.size(0), -1)), dim=1)

            x = F.relu(self.fc1_bn(x))
            x = self.dropout1(x)
            x = self.fc2(x)
            x = F.relu(self.fc2_bn(x))
            out = self.fc3(x)
        else:
            # normal LSTM
            out = self.fc3(out_lstm[:, -1, :])
        return out