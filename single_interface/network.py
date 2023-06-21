"""
	This file contains a neural network module for us to
	define our actor and critic networks in PPO.
"""

import torch
from torch import nn
import numpy as np


class SingleHeadRNN(nn.Module):
    def __init__(self, in_dim, tr_count, rnn_dim=5, hidden_dim=10):
        """
			Initialize the network and set up the layers.
			Return:
				None
		"""
        super(SingleHeadRNN, self).__init__()
        self.lstm = nn.LSTM(in_dim, rnn_dim, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(rnn_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, tr_count, bias=True)
        )

        self.rnn_dim = rnn_dim

    def init_hidden(self, batch_size=1):
        hidden_state = (torch.zeros(1, batch_size, self.rnn_dim),
                        torch.zeros(1, batch_size, self.rnn_dim))

        return hidden_state

    def reset_hidden(self, hidden_state):
        hidden_state[0][:, :, :] = 0
        hidden_state[1][:, :, :] = 0
        return hidden_state[0].detach(), hidden_state[1].detach()

    def forward(self, obs, hidden_state):
        """
			Runs a forward pass on the neural network.
		"""
        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        lstm_out, next_hidden_state = self.lstm(obs, hidden_state)
        final_output = self.mlp(lstm_out)
        return final_output, next_hidden_state
