import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class HashtagDataset(Dataset):
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path, lineterminator='\n')
        self.tweets = self.df.Tweet.tolist()
        self.labels = self.df.Stance.tolist()
        self.topics = self.df.Topic.tolist()

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {"tweets": self.tweets[idx], "labels": self.labels[idx], "topics": self.topics[idx]}

def batch_bert_tokenize(dataset_batch, tokenizer): 
    tokenized_inputs = tokenizer(dataset_batch["tweets"], max_length=64, padding="max_length", truncation=True, return_tensors='pt')
    dataset_batch['input_ids'] = tokenized_inputs['input_ids'][0]
    dataset_batch['attention_mask'] = tokenized_inputs['attention_mask'][0]
    return dataset_batch

def get_dataloaders_bert(train_path, val_path, test_path, batch_size):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    train = HashtagDataset(train_path)
    train_tokenized = train.map(lambda batch: batch_bert_tokenize(batch, tokenizer))
    train_loader = torch.utils.data.DataLoader(train_tokenized, batch_size=batch_size, drop_last=True, shuffle=True)

    val = HashtagDataset(val_path)
    val_tokenized = val.map(lambda batch: batch_bert_tokenize(batch, tokenizer))
    val_loader = torch.utils.data.DataLoader(val_tokenized, batch_size=batch_size, drop_last=True, shuffle=True)

    test = HashtagDataset(test_path)
    test_tokenized = test.map(lambda batch: batch_bert_tokenize(batch, tokenizer))
    test_loader = torch.utils.data.DataLoader(test_tokenized, batch_size=batch_size, drop_last=True, shuffle=True)

    return train_loader, val_loader, test_loader