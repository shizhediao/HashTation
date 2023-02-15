import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import datasets

class TweetDataset(Dataset):
    def __init__(self, data_path, fusion_type, low_resource, is_pilot, is_train):
        self.df = pd.read_csv(data_path, lineterminator='\n')
        if is_train and (is_pilot is not "none"):
            has_hashtags = []
            for s in self.df['text'].tolist():
                hashtag_cnt = s.count('#')
                has_hashtags.append(hashtag_cnt)
            self.df['hashtag_cnt'] = has_hashtags
            if is_pilot=="with":
                curr_df = self.df[self.df['hashtag_cnt']>=1]
            if is_pilot=="without":
                curr_df = self.df[self.df['hashtag_cnt']==0]
            self.df = curr_df.drop(['hashtag_cnt'], axis=1).reset_index(drop=True)
            _, self.df = train_test_split(self.df, test_size=100, stratify=self.df['label'])
            
            lens = 0
            for i in self.df['text'].tolist():
                lens += len(' '.join(i.strip().split()))
            print("Average length: ", lens / 100)
        elif is_train:
            if low_resource:
                sample_ratio = 0.1 if len(self.df) < 5000 else (0.05 if len(self.df) < 20000 else 0.01)
                _, self.df = train_test_split(self.df, test_size=sample_ratio, stratify=self.df['label'])
        self.df = self.df.dropna().reset_index(drop=True)
        self.tweets = self.df.text.tolist()
        self.labels = self.df.label.tolist()
        if "Generated_Hashtags" in self.df:
            self.hashtags = self.df["Generated_Hashtags"].tolist()
        if fusion_type != "none":
            assert("Generated_Hashtags" in self.df)
            self.fusion(fusion_type)
        print(len(self.tweets))

    def fusion(self, fusion_type):
        self.fused_topics = []
        if fusion_type=="standard":
            for tweet, hashtag in zip(self.tweets, self.hashtags):
                tweet_split = tweet.split()
                hashtag_split = hashtag.split('|')
                for i in range(len(tweet_split)):
                    if tweet_split[i] in hashtag_split:
                        tweet_split[i] = '#'+tweet_split[i]
                for i in range(len(hashtag_split)):
                    hashtag_split[i] = '#'+hashtag_split[i]
                for h in hashtag_split:
                    if h not in tweet_split:
                        tweet_split.append(h)
                self.fused_topics.append(' '.join(tweet_split))
        elif fusion_type=="start":
            for tweet, hashtag in zip(self.tweets, self.hashtags):
                tweet_split = tweet.split()
                hashtag_split = hashtag.split('|')
                for i in range(len(hashtag_split)):
                    hashtag_split[i] = '#'+hashtag_split[i]
                hashtag_split.extend(tweet_split)
                self.fused_topics.append(' '.join(hashtag_split))
        elif fusion_type=="end":
            for tweet, hashtag in zip(self.tweets, self.hashtags):
                tweet_split = tweet.split()
                hashtag_split = hashtag.split('|')
                for i in range(len(hashtag_split)):
                    hashtag_split[i] = '#'+hashtag_split[i]
                tweet_split.extend(hashtag_split)
                self.fused_topics.append(' '.join(tweet_split))

        # self.tweets = self.fused_topics
        return self.tweets

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {"tweets": self.tweets[idx], "labels": self.labels[idx]}

def batch_bert_tokenize(dataset_batch, tokenizer): 
    tokenized_inputs = tokenizer(dataset_batch["tweets"], max_length=128, padding="max_length", truncation=True, return_tensors='pt')
    dataset_batch['input_ids'] = tokenized_inputs['input_ids'][0]
    dataset_batch['attention_mask'] = tokenized_inputs['attention_mask'][0]
    return dataset_batch

def get_dataloaders_tweets(train_path, val_path, test_path, args):
    batch_size, model, fusion_type = args.batch_size, args.model, args.fusion_type
    if model=="timelms":
        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-mar2022")
    elif model=="bertweet":
        tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-large')
    elif model=="bert":
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    elif model=="bert-large":
        tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')
    elif model=="roberta":
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    elif model=="roberta-large":
        tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    else:
        assert(model in ["timelms", "bertweet", "bert", "bert-large", "roberta", "roberta-large"])

    train = TweetDataset(train_path, fusion_type, args.low_resource, is_pilot=args.pilot, is_train=True)
    train_tokenized = train.map(lambda batch: batch_bert_tokenize(batch, tokenizer))
    train_loader = torch.utils.data.DataLoader(train_tokenized, batch_size=batch_size, drop_last=True, shuffle=True)

    val = TweetDataset(val_path, fusion_type, args.low_resource, is_pilot=args.pilot, is_train=False)
    val_tokenized = val.map(lambda batch: batch_bert_tokenize(batch, tokenizer))
    val_loader = torch.utils.data.DataLoader(val_tokenized, batch_size=batch_size, drop_last=True, shuffle=True)

    test = TweetDataset(test_path, fusion_type, args.low_resource, is_pilot=args.pilot, is_train=False)
    test_tokenized = test.map(lambda batch: batch_bert_tokenize(batch, tokenizer))
    test_loader = torch.utils.data.DataLoader(test_tokenized, batch_size=batch_size, drop_last=True, shuffle=True)

    return train_loader, val_loader, test_loader


class HashtagDataset(Dataset):
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path, lineterminator='\n')
        self.tweets = self.df.text.tolist()
        self.labels = self.df.label.tolist()
        self.hashtags = self.df.hashtags.tolist()

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {"tweets": self.tweets[idx], "labels": self.labels[idx], "hashtags": self.hashtags[idx]}

def batch_bert_tokenize_hashtags(dataset_batch, tokenizer): 
    tokenized_inputs = tokenizer(dataset_batch["text"], max_length=128, padding="max_length", truncation=True, return_tensors='pt')
    dataset_batch['input_ids'] = tokenized_inputs['input_ids'][0]
    dataset_batch['attention_mask'] = tokenized_inputs['attention_mask'][0]
    tokenized_inputs_hashtags = tokenizer(dataset_batch["hashtags"], max_length=64, padding="max_length", truncation=True, return_tensors='pt')
    dataset_batch['hashtag_input_ids'] = tokenized_inputs_hashtags['input_ids'][0]
    dataset_batch['hashtag_attention_mask'] = tokenized_inputs_hashtags['attention_mask'][0]
    return dataset_batch

def get_dataloaders_hashtags(train_path, val_path, test_path, batch_size, model):
    if model == "bart-base":
        tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
    elif model == "bart-large":
        tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')
    elif model == "kp-times":
        tokenizer = AutoTokenizer.from_pretrained('ankur310794/bart-base-keyphrase-generation-kpTimes')
    else:
        tokenizer = None

    train = datasets.Dataset.from_csv(train_path)
    train_tokenized = train.map(lambda batch: batch_bert_tokenize_hashtags(batch, tokenizer))
    train_loader = torch.utils.data.DataLoader(train_tokenized, batch_size=batch_size, drop_last=True, shuffle=True)

    val = datasets.Dataset.from_csv(val_path)
    val_tokenized = val.map(lambda batch: batch_bert_tokenize_hashtags(batch, tokenizer))
    val_loader = torch.utils.data.DataLoader(val_tokenized, batch_size=batch_size, drop_last=True, shuffle=True)

    test = datasets.Dataset.from_csv(test_path)
    test_tokenized = test.map(lambda batch: batch_bert_tokenize_hashtags(batch, tokenizer))
    test_loader = torch.utils.data.DataLoader(test_tokenized, batch_size=batch_size, drop_last=True, shuffle=True)

    return train_loader, val_loader, test_loader