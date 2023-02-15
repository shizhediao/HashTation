import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel

class KimCNN(nn.Module):
    def __init__(self, num_classes, vocab_size, in_channels, out_channels, linear_size, kernel_size, net_dropout):
        super(KimCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, 768)
        self.dropout = nn.Dropout(net_dropout)
        self.linear = nn.Linear(out_channels*len(kernel_size), num_classes)
        
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=K) for K in kernel_size])

    def forward(self, x, x_len, topic):
        out = self.embedding(x).transpose(1,2)      # [bsz, seq_len, emb_size]
        conv_outs = [F.relu(conv(out)) for conv in self.convs]      # [bsz, out_filters, seq_len-kernel_size]
        pooled_outs = [F.max_pool1d(x, kernel_size=x.shape[2]).squeeze(2) for x in conv_outs]      # [bsz, out_filters]
        
        out = torch.cat(pooled_outs, 1)
        out = self.dropout(out)
        out = self.linear(out)
        return out