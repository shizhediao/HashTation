import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

class TAM_Module(nn.Module):
    def __init__(self, train_path, model, emb_module, device):
        super().__init__()
        self.train_path = train_path
        self.device = device
        self.emb_module = emb_module.to(device)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        if model=="bart-base":
            self.model_dim = 768
        elif model=="bart-large":
            self.model_dim = 1024
        self.linear = nn.Linear(384, self.model_dim).to(self.device)
        self.init_sentence_embedding_matrix()

    def init_sentence_embedding_matrix(self):
        self.df = pd.read_csv(self.train_path, lineterminator='\n')
        self.tweets = self.df.text.tolist()
        self.sent_embeddings = nn.Parameter(torch.Tensor(self.model.encode(self.tweets)), requires_grad=True).to(self.device)

    def forward(self, sent, input_ids):
        embs = self.emb_module(input_ids)
        curr_embedding = torch.Tensor(self.model.encode(sent)).to(self.device)
        weights = self.sent_embeddings@curr_embedding.T
        weights = torch.nn.functional.softmax(weights, dim=0)
        o = weights.T@self.sent_embeddings
        o = self.linear(o)
        return embs + o.unsqueeze(1)
