import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import AutoModel

class BiLSTM(nn.Module):
    def __init__(self, num_classes, vocab_size, linear_size, lstm_hidden_size, net_dropout, lstm_dropout):

        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 768)
        self.dropout = nn.Dropout(net_dropout)
        
        self.hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(768, self.hidden_size, dropout=lstm_dropout, bidirectional=True)
        self.linear = nn.Linear(self.hidden_size*2, linear_size)
        self.out = nn.Linear(linear_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x, x_len, topic): 
        embs = self.embedding(x)
        seq_lengths, perm_idx = x_len.sort(0, descending=True)
        seq_tensor = embs[perm_idx]
        packed_input = pack_padded_sequence(seq_tensor, seq_lengths, batch_first=True)
        packed_output, (ht, ct) = self.lstm(packed_input)
        _, unperm_idx = perm_idx.sort(0)
        h_t = ht[:,unperm_idx,:]
        h_t = torch.cat((h_t[0,:,:self.hidden_size], h_t[1,:,:self.hidden_size]), 1)
        
        linear = self.relu(self.linear(h_t))
        linear = self.dropout(linear)
        out = self.out(linear)
        return out
