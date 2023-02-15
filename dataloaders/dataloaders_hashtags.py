import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class HashtagDataset(Dataset):
    def __init__(self, data_path, fusion_type):
        self.df = pd.read_csv(data_path, lineterminator='\n')
        self.tweets = self.df.Tweet.tolist()
        self.labels = self.df.Stance.tolist()
        self.hashtags = self.df.Generated_Hashtags.tolist()
        self.topics = self.df.Topic.tolist()
        self.fusion(fusion_type)

    def fusion(self, fusion_type):
        self.fused_topics = []
        for tweet, hashtag in zip(self.tweets, self.hashtags):
            hashtag_split = hashtag.split(',')
            grouped_hashtags = ""
            for h in hashtag_split: 
                grouped_hashtags += f"#{h} "
            if fusion_type=="start":
                self.fused_topics.append(f"{hashtag.strip()} {tweet}")
            elif fusion_type=="end":
                self.fused_topics.append(f"{tweet} {hashtag.strip()}")
            elif fusion_type=="about":
                self.fused_topics.append(f"About {hashtag.strip()}: {tweet}")
            elif fusion_type=="none":
                self.fused_topics.append(tweet)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {"tweets": self.fused_topics[idx], "labels": self.labels[idx], "hashtags": self.hashtags[idx], "topics": self.topics[idx]}

def batch_bert_tokenize(dataset_batch, tokenizer): 
    tokenized_inputs = tokenizer(dataset_batch["tweets"], max_length=64, padding="max_length", truncation=True, return_tensors='pt')
    dataset_batch['input_ids'] = tokenized_inputs['input_ids'][0]
    dataset_batch['attention_mask'] = tokenized_inputs['attention_mask'][0]
    return dataset_batch

def get_dataloaders_hashtags(train_path, val_path, test_path, batch_size, fusion_type):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    train = HashtagDataset(train_path, fusion_type)
    train_tokenized = train.map(lambda batch: batch_bert_tokenize(batch, tokenizer))
    train_loader = torch.utils.data.DataLoader(train_tokenized, batch_size=batch_size, drop_last=True, shuffle=True)

    val = HashtagDataset(val_path, fusion_type)
    val_tokenized = val.map(lambda batch: batch_bert_tokenize(batch, tokenizer))
    val_loader = torch.utils.data.DataLoader(val_tokenized, batch_size=batch_size, drop_last=True, shuffle=True)

    test = HashtagDataset(test_path, fusion_type)
    test_tokenized = test.map(lambda batch: batch_bert_tokenize(batch, tokenizer))
    test_loader = torch.utils.data.DataLoader(test_tokenized, batch_size=batch_size, drop_last=True, shuffle=True)

    return train_loader, val_loader, test_loader