import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, process_results

from utils import set_logger
from models import *
from dataloaders.dataloaders_baselines import get_dataloaders_baseline_new

def get_model(model_name, vocab_size, args):
    assert(model_name in ["BiLSTM", "GCAE", "KimCNN"])
    num_class_map = {"emoji":20, "emotion":4, "hate":2, "irony":2, "offensive":2, "sentiment":3, "stance":3}
    num_classes = num_class_map[args.dataset]
    if args.model_name == "BiLSTM":
        model = BiLSTM(num_classes=num_classes, vocab_size=vocab_size, linear_size=32, lstm_hidden_size=512, net_dropout=0.5, lstm_dropout=0.2)
    elif args.model_name == "GCAE":
        model = GCAE(num_classes=num_classes, vocab_size=vocab_size, kernel_num=100, kernel_sizes=[3,4,5], aspect_kernel_num=100, aspect_kernel_sizes=[3], dropout=0.2)
    elif args.model_name == "KimCNN":
        model = KimCNN(num_classes=num_classes, vocab_size=vocab_size, in_channels=768, out_channels=64, linear_size=128, kernel_size=[2,3,4,5], net_dropout=args.dropout)
    else:
        model = None
    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_path = f"tweeteval-processed-full/{args.dataset}/train.csv"
    val_path = f"tweeteval-processed-full/{args.dataset}/val.csv"
    test_path = f"tweeteval-processed-full/{args.dataset}/test.csv"
    train_loader, val_loader, test_loader, vocab_size = get_dataloaders_baseline_new(train_path, val_path, test_path, args.batch_size, args.seq_len)

    # Initialize model, optimizer, loss
    model = get_model(args.model_name, vocab_size, args)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    for e in range(args.n_epochs):
        model.train()
        train_preds, train_labels, train_loss = [], [], 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            x, labels, xlen, topic = batch["x"].to(device), batch["y"].to(device), batch["xlen"], batch["topic"].to(device)
            outputs = model(x, xlen, topic)
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
            x, labels, xlen, topic = batch["x"].to(device), batch["y"].to(device), batch["xlen"], batch["topic"].to(device)
            outputs = model(x, xlen, topic)
            loss = loss_func(outputs, labels)

            val_preds.extend(outputs.argmax(axis=1).cpu().detach().numpy())
            val_labels.extend(labels.cpu().detach().numpy())
            val_loss += loss.item()
        acc = sum(1 for x,y in zip(val_preds, val_labels) if x == y) / len(val_labels)
        logging.info(f"Valid epoch {e}: Loss={val_loss/len(val_loader)} | Acc={acc}")

    # Testing
    model.eval()
    test_preds, test_labels, test_loss = [], [], 0
    for batch in tqdm(test_loader):
        x, labels, xlen, topic = batch["x"].to(device), batch["y"].to(device), batch["xlen"], batch["topic"].to(device)
        outputs = model(x, xlen, topic)
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


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="BiLSTM", type=str, choices=["BiLSTM", "KimCNN"])
    parser.add_argument('--dataset', type=str, required=True, choices=["emoji", "emotion", "hate", "irony", "offensive", "sentiment", "stance"])

    parser.add_argument('--seq_len', default=64, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--n_epochs', default=20, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--logging', action='store_true')

    args = parser.parse_args()
    set_logger(args.logging)    
    logging.info(f"args: {args}")

    main(args)