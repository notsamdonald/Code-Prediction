# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Code_GPT_modified(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(Code_GPT_modified, self).__init__()
        self.ntoken = ntoken
        self.nlayers = nlayers
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, batch_first=True)
        self.GAE_input = nn.LSTM(8, 768, nlayers) # TODO - configure correctly
        self.decoder_1 = nn.Linear(nhid, int(ntoken/3))
        self.decoder_2= nn.Linear(int(ntoken/3), ntoken)
        self.criterion = nn.CrossEntropyLoss()

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder_1.weight)
        nn.init.zeros_(self.decoder_2.weight)
        nn.init.uniform_(self.decoder_1.weight, -initrange, initrange)
        nn.init.uniform_(self.decoder_2.weight, -initrange, initrange)

    def forward(self, input, GAE_embedding=None, labels=None, target=None):
        emb = self.encoder(input)
        if GAE_embedding is not None:
            output, (h,c) = self.GAE_input(GAE_embedding.reshape(-1,8))
            h = h.reshape(self.nlayers,1,-1)
            c = torch.zeros(h.shape)
            output, hidden = self.rnn(emb, (h,c))  # FIXME - C = 0 here, only h coppied over
        else:
            output, hidden = self.rnn(emb)

        if target is not None:
            output = hidden[0]
            # output = self.drop(hidden[0])

        output = self.drop(output)
        output = self.decoder_1(output)
        output = self.drop(output)
        output = self.decoder_2(output)

        # decoded = decoded.view(-1, self.ntoken)
        # output = F.log_softmax(decoded, dim=1)

        if labels is not None and target is None:
            shift_logits = output[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss, output, hidden

        elif target is not None:
            loss = self.criterion(output.reshape(1, -1), target.reshape(1))
            return loss, output

        else:
            return output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

