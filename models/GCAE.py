# GCAE Model
# Paper "Aspect Based Sentiment Analysis with Gated Convolutional Networks"
# https://www.aclweb.org/anthology/P18-1234.pdf
# Implementation from https://github.com/jiangqn/GCAE-pytorch

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel


class GCAE(nn.Module):
    # def __init__(self, num_classes, vocab_size, kernel_num, kernel_size):

    def __init__(self, num_classes, vocab_size, kernel_num, kernel_sizes, aspect_kernel_num, aspect_kernel_sizes, dropout):
        super(GCAE, self).__init__()
        self._embedding = nn.Embedding(vocab_size, 768)
        embed_size = 768
        aspect_embed_size = 768
        self._sentence_conv = nn.ModuleList(
            nn.Conv1d(
                in_channels=embed_size,
                out_channels=kernel_num,
                kernel_size=kernel_size
            ) for kernel_size in kernel_sizes
        )
        self._sentence_conv_gate = nn.ModuleList(
            nn.Conv1d(
                in_channels=embed_size,
                out_channels=kernel_num,
                kernel_size=kernel_size
            ) for kernel_size in kernel_sizes
        )
        self._aspect_conv = nn.ModuleList(
            nn.Conv1d(
                in_channels=aspect_embed_size,
                out_channels=aspect_kernel_num,
                kernel_size=aspect_kernel_size,
                padding=1
            ) for aspect_kernel_size in aspect_kernel_sizes
        )
        self._aspect_linear = nn.Linear(len(aspect_kernel_sizes) * aspect_kernel_num, kernel_num)
        self._dropout = nn.Dropout(dropout)
        self._linear = nn.Linear(len(kernel_sizes) * kernel_num, num_classes)

    def forward(self, sentence, _, aspect):
        # sentence: Tensor (batch_size, sentence_length)
        # aspect: Tensor (batch_size, aspect_length)
        sentence = self._embedding(sentence)
        aspect = self._embedding(aspect)
        aspect = torch.cat([
            torch.max(
                F.relu(conv(aspect.transpose(1, 2))),
                dim=-1
            )[0] for conv in self._aspect_conv
        ], dim=1)
        aspect = self._aspect_linear(aspect)
        sentence = torch.cat([
            torch.max(
                torch.tanh(conv(sentence.transpose(1, 2))) * F.relu(conv_gate(sentence.transpose(1, 2)) + aspect.unsqueeze(2)),
                dim=-1
            )[0] for conv, conv_gate in zip(self._sentence_conv, self._sentence_conv_gate)
        ], dim=1)
        sentence = self._dropout(sentence)
        logit = self._linear(sentence)
        return logit