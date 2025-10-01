import torch
import torch.nn as nn

class BaseRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=300, hidden_size=256, num_layers=1, bidirectional=True, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

    def forward(self, *args, **kwargs):
        raise NotImplementedError

class RNNModel(BaseRNN):
    def __init__(self, vocab_size, **kwargs):
        super().__init__(vocab_size, **kwargs)
        self.rnn = nn.RNN(self.embedding.embedding_dim, self.hidden_size, self.num_layers, batch_first=True, bidirectional=self.bidirectional)
        self.classifier = nn.Linear(self.hidden_size * (2 if self.bidirectional else 1), 2)  # start/end logits

    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        out, _ = self.rnn(emb)
        logits = self.classifier(out) # shape: (B, T, 2)
        start_logits = logits[...,0]
        end_logits = logits[...,1]
        return start_logits, end_logits

class GRUModel(BaseRNN):
    def __init__(self, vocab_size, **kwargs):
        super().__init__(vocab_size, **kwargs)
        self.rnn = nn.GRU(self.embedding.embedding_dim, self.hidden_size, self.num_layers, batch_first=True, bidirectional=self.bidirectional)
        self.classifier = nn.Linear(self.hidden_size * (2 if self.bidirectional else 1), 2)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        out, _ = self.rnn(emb)
        logits = self.classifier(out)
        return logits[...,0], logits[...,1]

class LSTMModel(BaseRNN):
    def __init__(self, vocab_size, **kwargs):
        super().__init__(vocab_size, **kwargs)
        self.rnn = nn.LSTM(self.embedding.embedding_dim, self.hidden_size, self.num_layers, batch_first=True, bidirectional=self.bidirectional)
        self.classifier = nn.Linear(self.hidden_size * (2 if self.bidirectional else 1), 2)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        out, _ = self.rnn(emb)
        logits = self.classifier(out)
        return logits[...,0], logits[...,1]
