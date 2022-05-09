import torch
import torch.nn as nn


# Based on https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/CodeCompletion-token/code/model.py

class LSTM_predictor(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=True):
        super(LSTM_predictor, self).__init__()
        self.ntoken = ntoken
        self.nlayers = nlayers
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, batch_first=True)
        self.GAE_input = nn.LSTM(8, 768, nlayers)  # TODO - configure correctly
        self.decoder = nn.Linear(nhid, ntoken)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.CrossEntropyLoss()

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
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, GAE_embedding=None, labels=None, target=None):
        emb = self.encoder(input)
        if GAE_embedding is not None:
            output, (h, c) = self.GAE_input(GAE_embedding.reshape(-1, 8))
            h = h.reshape(self.nlayers, 1, -1).to(self.device)
            c = torch.zeros(h.shape).to(self.device)
            output, hidden = self.rnn(emb, (h, c))  # FIXME - C = 0 here, only h used
        else:
            output, hidden = self.rnn(emb)

        if target is not None:
            output = hidden[0]

        output = self.drop(output)
        output = self.decoder(output)

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
