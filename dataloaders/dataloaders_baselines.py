import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class HashtagDataset(Dataset):
    def __init__(self, data_path, text_pipeline, seq_len):
        self.df = pd.read_csv(data_path, lineterminator='\n')
        self.tweets = self.df.Tweet.tolist()
        self.tweets_emb = [text_pipeline(t) for t in self.tweets]
        self.dataset = data_path.split('/')[-2]
        
        self.lens = [len(t) if len(t)<seq_len else seq_len for t in self.tweets_emb]
        self.x = torch.tensor([self.pad_or_cut(t, seq_len) for t in self.tweets_emb])
        self.y = torch.tensor(self.df.Stance.tolist())
        if self.dataset=="semeval2016t6":
            # Convert all to two words
            self.df["Topic"] = self.df["Topic"].map({'Legalization of Abortion':"Abortion Legalization", 
            "Climate Change is a Real Concern":"Climate Change", 
            "Atheism":"Atheism Atheism",
            "Donald Trump":"Donald Trump",
            "Hillary Clinton":"Hillary Clinton",
            "Feminist Movement":"Feminist Movement"})
        self.topics = torch.tensor([text_pipeline(t) for t in self.df.Topic.tolist()])

    def pad_or_cut(self, arr, seq_len, pad_idx=1):
        if len(arr) >= seq_len:
            return arr[:seq_len]
        else:
            while len(arr) < seq_len:
                arr.append(pad_idx)
            return arr

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return {"x": self.x[idx], "y": self.y[idx], "xlen": self.lens[idx], "topic": self.topics[idx]}

def set_tokenizer_vocab(train_path):
    train_df = pd.read_csv(train_path,lineterminator='\n')
    train_iter = train_df.Tweet.tolist() + (list(set(train_df.Topic.tolist())))
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>', '<pad>'])      #<unk>=0, <pad>=1
    vocab.set_default_index(vocab['<unk>'])
    text_pipeline = lambda x: vocab(tokenizer(x))
    return text_pipeline, len(vocab)

def get_dataloaders_baseline(train_path, val_path, test_path, batch_size, seq_len):
    text_pipeline, vocab_size = set_tokenizer_vocab(train_path)
    train_dataset = HashtagDataset(train_path, text_pipeline, seq_len)
    val_dataset = HashtagDataset(val_path, text_pipeline, seq_len)
    test_dataset = HashtagDataset(test_path, text_pipeline, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

    return train_loader, val_loader, test_loader, vocab_size


### For Tweeteval
class HashtagDataset_v2(Dataset):
    def __init__(self, data_path, text_pipeline, seq_len):
        self.df = pd.read_csv(data_path, lineterminator='\n')
        self.tweets = self.df.text.tolist()
        self.tweets_emb = [text_pipeline(t) for t in self.tweets]
        self.dataset = data_path.split('/')[-2]
        
        self.lens = [len(t) if len(t)<seq_len else seq_len for t in self.tweets_emb]
        self.x = torch.tensor([self.pad_or_cut(t, seq_len) for t in self.tweets_emb])
        self.y = torch.tensor(self.df.label.tolist())
        self.topics = [0 for i in self.y]       # useless; just a placeholder

    def pad_or_cut(self, arr, seq_len, pad_idx=1):
        if len(arr) >= seq_len:
            return arr[:seq_len]
        else:
            while len(arr) < seq_len:
                arr.append(pad_idx)
            return arr

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return {"x": self.x[idx], "y": self.y[idx], "xlen": self.lens[idx], "topic": self.topics[idx]}

def set_tokenizer_vocab_new(train_path):
    train_df = pd.read_csv(train_path,lineterminator='\n')
    train_iter = train_df.text.tolist() + (list(set(train_df.text.tolist())))
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>', '<pad>'])      #<unk>=0, <pad>=1
    vocab.set_default_index(vocab['<unk>'])
    text_pipeline = lambda x: vocab(tokenizer(x))
    return text_pipeline, len(vocab)

def get_dataloaders_baseline_new(train_path, val_path, test_path, batch_size, seq_len):
    text_pipeline, vocab_size = set_tokenizer_vocab_new(train_path)
    train_dataset = HashtagDataset_v2(train_path, text_pipeline, seq_len)
    val_dataset = HashtagDataset_v2(val_path, text_pipeline, seq_len)
    test_dataset = HashtagDataset_v2(test_path, text_pipeline, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

    return train_loader, val_loader, test_loader, vocab_size