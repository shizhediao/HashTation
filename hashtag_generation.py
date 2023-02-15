import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, BartForConditionalGeneration, EncoderDecoderModel

from utils import set_logger
from dataloaders.dataloaders_classification import get_dataloaders_hashtags
from module_tam import TAM_Module

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model=="bart-base":
        bart_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
    elif args.model=="bart-large":
        bart_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')
    elif args.model=="kp-times":
        bart_tokenizer = AutoTokenizer.from_pretrained('ankur310794/bart-base-keyphrase-generation-kpTimes')
    else:
        bart_tokenizer = None

    # Load data
    train_path = f"tweeteval-hashtags/{args.dataset}/train.csv"
    val_path = f"tweeteval-hashtags/{args.dataset}/val.csv"
    test_path = f"tweeteval-hashtags/{args.dataset}/test.csv"
    train_loader, val_loader, test_loader = get_dataloaders_hashtags(train_path, val_path, test_path, args.batch_size, args.model)

    # Initialize models, optimizer, loss
    if args.model=="bart-base":
        bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    elif args.model=="bart-large":
        bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    elif args.model=="kp-times":
        bart_model = BartForConditionalGeneration.from_pretrained('ankur310794/bart-base-keyphrase-generation-kpTimes')
    else:
        bart_model = None
    bart_model.to(device)
    optimizer = torch.optim.Adam(bart_model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    if args.tam_module:
        emb_module = bart_model.model.shared
        tam = TAM_Module(train_path, args.model, emb_module, device)

    for e in range(args.n_epochs):
        bart_model.train()
        train_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            inputs_embeds, input_ids, attention_mask = None, torch.stack(batch["input_ids"]).transpose(0,1).contiguous().to(device), torch.stack(batch["attention_mask"]).transpose(0,1).contiguous().to(device)
            hashtag_input_ids, labels =  torch.stack(batch["hashtag_input_ids"]).transpose(0,1).contiguous().to(device), None
            if args.tam_module:
                inputs_embeds, input_ids = tam(batch["text"], input_ids), None

            loss = bart_model(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=hashtag_input_ids)["loss"]
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        logging.info(f"Train epoch {e}: Loss={train_loss/len(train_loader)}")

        bart_model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                inputs_embeds, input_ids, attention_mask = None, torch.stack(batch["input_ids"]).transpose(0,1).contiguous().to(device), torch.stack(batch["attention_mask"]).transpose(0,1).contiguous().to(device)
                hashtag_input_ids, labels =  torch.stack(batch["hashtag_input_ids"]).transpose(0,1).contiguous().to(device), None     
                if args.tam_module:
                    inputs_embeds, input_ids = tam(batch["text"], input_ids), None
                
                loss = bart_model(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=hashtag_input_ids)["loss"]
                generated_hashtags = bart_model.generate(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                    length_penalty=args.length_penalty, num_beams=args.beam_size,
                    min_length=args.decoder_min_length, max_length=64,
                    early_stopping=args.decoder_early_stopping, no_repeat_ngram_size=args.no_repeat_ngram_size)
                generated_hashtags = [bart_tokenizer.decode(h, skip_special_tokens=True) for h in generated_hashtags]
                tweets = [f"{h} | {h2} | {t}" for h,h2,t in zip(generated_hashtags, batch["hashtags"], batch["text"])]
                val_loss += loss.item()
            print(tweets)
        logging.info(f"Valid epoch {e}: Loss={val_loss/len(val_loader)}")

    # Testing
    bart_model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            inputs_embeds, input_ids, attention_mask = None, torch.stack(batch["input_ids"]).transpose(0,1).contiguous().to(device), torch.stack(batch["attention_mask"]).transpose(0,1).contiguous().to(device)
            hashtag_input_ids, labels =  torch.stack(batch["hashtag_input_ids"]).transpose(0,1).contiguous().to(device), None   
            if args.tam_module:
                inputs_embeds, input_ids = tam(batch["text"], input_ids), None
            
            loss = bart_model(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=hashtag_input_ids)["loss"]
            generated_hashtags = bart_model.generate(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                length_penalty=args.length_penalty, num_beams=args.beam_size,
                min_length=args.decoder_min_length, max_length=64,
                early_stopping=args.decoder_early_stopping, no_repeat_ngram_size=args.no_repeat_ngram_size)
            generated_hashtags = [bart_tokenizer.decode(h, skip_special_tokens=True) for h in generated_hashtags]
            tweets = [f"{h} | {h2} | {t}" for h,h2,t in zip(generated_hashtags, batch["hashtags"], batch["text"])]
            test_loss += loss.item()
    logging.info(f"Test: Loss={test_loss/len(test_loader)}")

    # Save model
    if args.save_path != "none":
        if args.tam_module:
            torch.save({'bart_model_state_dict': bart_model.state_dict(),
                        'tam_state_dict': tam.state_dict()}, args.save_path)
        else:
            torch.save({'bart_model_state_dict': bart_model.state_dict()}, args.save_path)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=["emoji", "emotion", "hate", "irony", "offensive", "sentiment", "stance", "unified", "all"])
    parser.add_argument('--model', type=str, default='kp-times', choices=["bart-base", "bart-large", "kp-times"])

    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--logging', action='store_true')
    parser.add_argument('--save_path', type=str, default="none")

    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--decoder_early_stopping', default=True, type=bool)
    parser.add_argument('--length_penalty', default=0.6, type=float)
    parser.add_argument('--decoder_min_length', default=1, type=int)
    parser.add_argument('--no_repeat_ngram_size', default=3, type=int)

    parser.add_argument('--tam_module', action='store_true')

    args = parser.parse_args()
    set_logger(args.logging)    
    logging.info(f"args: {args}")

    main(args)