import torch
import torch.nn as nn



rnn = nn.LSTM(8, 50, 1)
input = torch.randn(239, 8)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(input, (h0, c0))
print("")