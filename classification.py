import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from transformers import AutoTokenizer

from utils import set_logger, process_results
from models import ClassificationModel
from dataloaders.dataloaders_classification import get_dataloaders_tweets

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders_tweets(args.train_path, args.val_path, args.test_path, args)

    # Initialize models, optimizer, loss
    num_class_map = {"emoji":20, "emotion":4, "hate":2, "irony":2, "offensive":2, "sentiment":3, "stance":3}
    model = ClassificationModel(num_class_map[args.dataset], args.frozen)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    for e in range(args.n_epochs):
        model.train()
        train_preds, train_labels, train_loss = [], [], 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].long().to(device)
            outputs = model(input_ids, attention_mask)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            train_preds.extend(outputs.argmax(axis=1).cpu().detach().numpy())
            train_labels.extend(labels.cpu().detach().numpy())
            train_loss += loss.item()
        acc = sum(1 for x,y in zip(train_labels, train_preds) if x == y) / len(train_labels)
        logging.info(f"Train epoch {e}: Loss={train_loss/len(train_loader)} | Acc={acc}")

        model.eval()
        val_preds, val_labels, val_loss = [], [], 0
        for batch in tqdm(val_loader):
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].long().to(device)
            outputs = model(input_ids, attention_mask)
            loss = loss_func(outputs, labels)

            val_preds.extend(outputs.argmax(axis=1).cpu().detach().numpy())
            val_labels.extend(labels.cpu().detach().numpy())
            val_loss += loss.item()
        # acc = sum(1 for x,y in zip(val_preds, val_labels) if x == y) / len(val_labels)
        results = classification_report(val_labels, val_preds, digits=4)
        results_dict = classification_report(val_labels, val_preds, digits=4, output_dict=True)
        logging.info(f"Valid epoch {e}: Loss={val_loss/len(val_loader)} | F1={process_results(results_dict, args.dataset)}")

    # Testing
    model.eval()
    test_preds, test_labels, test_loss = [], [], 0
    for batch in tqdm(test_loader):
        input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].long().to(device)
        outputs = model(input_ids, attention_mask)
        loss = loss_func(outputs, labels)

        test_preds.extend(outputs.argmax(axis=1).cpu().detach().numpy())
        test_labels.extend(labels.cpu().detach().numpy())
        test_loss += loss.item()
    acc = sum(1 for x,y in zip(test_preds, test_labels) if x == y) / len(test_labels)
    logging.info(f"Test: Loss={test_loss/len(test_loader)} | Acc={acc}")
    results = classification_report(test_labels, test_preds, digits=4)
    results_dict = classification_report(test_labels, test_preds, digits=4, output_dict=True)
    logging.info(results)
    logging.info(f"Final test result: {process_results(results_dict, args.dataset)}")
    if args.out_file is not None:
        with open(args.out_file, 'a') as f:
            f.write(str(process_results(results_dict, args.dataset))+'\n')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=["emoji", "emotion", "hate", "irony", "offensive", "sentiment", "stance"])
    parser.add_argument('--model', type=str, default='timelms', choices=["timelms", "bertweet", "bert", "roberta", "bert-large", "roberta-large"])
    parser.add_argument('--fusion_type', type=str, default="none", choices=["standard", "start", "end", "none"])
    parser.add_argument('--tam_module', action='store_true')
    parser.add_argument('--frozen', action='store_true')
    parser.add_argument('--low_resource', action='store_true')
    parser.add_argument('--pilot', type=str, default="none", choices=["with", "without", "none"])

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--logging', action='store_true')
    parser.add_argument('--out_file', type=str, default=None)

    args = parser.parse_args()
    set_logger(args.logging)    
    logging.info(f"args: {args}")

    tweetdir = "tweeteval-processed-full"
    if args.fusion_type=="none":
        args.train_path = f"{tweetdir}/{args.dataset}/train.csv"
        args.val_path = f"{tweetdir}/{args.dataset}/val.csv"
        args.test_path = f"{tweetdir}/{args.dataset}/test.csv"
    else:
        tweetdir = "tweeteval-hashtags-gen"
        args.train_path = f"{tweetdir}/{args.dataset}/train.csv"
        args.val_path = f"{tweetdir}/{args.dataset}/val.csv"
        args.test_path = f"{tweetdir}/{args.dataset}/test.csv"

    main(args)