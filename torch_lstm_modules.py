import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

import math

# 1. Bidirectional Encoder
# 2. Hierarchical Decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BidirectionalLSTMEncoder(nn.Module):
    def __init__(self,
                 input_size=42,
                 hidden_size=2048,
                 latent_size=512,
                 num_layer=2):
        super(BidirectionalLSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layer = num_layer

        self.bilstm = nn.LSTM(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layer=num_layer,
                              bidirectional=True)
        self.mu = nn.Linear(in_features=2 * hidden_size, out_features=latent_size)
        self.sigma = nn.Linear(in_features=2 * hidden_size, out_features=latent_size)
        self.softmax = nn.Softmax()

    def forward(self, input, h0, c0):
        batch_size = input.size(1)
        _, (h_n, c_n) = self.bilstm(input, (h0, c0))
        h_n = h_n.view(self.num_layers, 2, batch_size, -1)[-1].view(batch_size, -1)
        mu = self.mu(h_n)
        sigma = self.sigma(h_n)
        sigma = math.exp(sigma) + 1
        sigma = math.log(sigma, 10)
        return mu, sigma

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.num_layers*2, batch_size, self.hidden-size, dtype=torch.float, device=device),
                torch.zeros(self.num_layers*2, batch_size, self.hidden - size, dtype=torch.float, device=device))


class HierarchicalLSTMDecoder(nn.Module):
    def __init__(self,
                 num_embeddings,
                 input_size=42,
                 hidden_size=1024,
                 latent_size=512,
                 num_layers=2,
                 max_seq_length=256,
                 seq_length=16):
        super(HierarchicalLSTMDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_embeddings = num_embeddings
        self.max_seq_length = max_seq_length
        self.seq_length = seq_length
        self.num_layers = num_layers

        self.tanh = nn.Tanh()
        self.conductor = nn.LSTM(input_size=latent_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers)
        self.conductor_embeddings = nn.Sequential(
            nn.Linear(in_features=hidden_size,
                      out_features=latent_size),
            nn.Tanh())
        self.lstm = nn.LSTM(input_size=input_size + latent_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers)
        self.out = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=input_size),
            nn.Softmax(dim=2)
        )

    def forward(selfself, target, latent, h0, c0, use_teacher_forcing=True, temperature=1.0):
        pass
