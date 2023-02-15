import logging
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration
from module_tam import TAM_Module
from utils import set_logger

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

    # Initialize models, optimizer, loss
    if args.model=="bart-base":
        bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    elif args.model=="bart-large":
        bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    elif args.model=="kp-times":
        bart_model = BartForConditionalGeneration.from_pretrained('ankur310794/bart-base-keyphrase-generation-kpTimes')
    else:
        bart_model = None
    bart_model.load_state_dict(torch.load(args.model_path)['bart_model_state_dict'])
    bart_model.to(device)
    
    if args.tam_module:
        train_path = f"tweeteval-hashtags/{args.dataset}/train.csv"
        emb_module = bart_model.model.shared
        tam = TAM_Module(train_path, args.model, emb_module, device)
        tam.load_state_dict(torch.load(args.model_path)['tam_state_dict'])
        tam.to(device)

    # Load data
    for split in ["train", "val", "test"]:
        curr_path = f"tweeteval-processed-full/{args.dataset}/{split}.csv"
        data = pd.read_csv(curr_path, lineterminator='\n')
        generated_hashtags = []
        for tweet in tqdm(data["text"]):
            tok = bart_tokenizer(tweet, return_tensors='pt')
            inputs_embeds, input_ids, attention_mask = None, tok["input_ids"].to(device), tok["attention_mask"].to(device)
            if args.tam_module:
                inputs_embeds, input_ids = tam([tweet], input_ids), None
            curr_hashtags = bart_model.generate(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                length_penalty=args.length_penalty, num_beams=args.beam_size,
                min_length=args.decoder_min_length, max_length=64,
                early_stopping=args.decoder_early_stopping, no_repeat_ngram_size=args.no_repeat_ngram_size)
            generated_hashtags.append(bart_tokenizer.decode(curr_hashtags[0], skip_special_tokens=True))
        data["Generated_Hashtags"] = generated_hashtags
        dataset_name = args.dataset if not args.tam_module else f"{args.dataset}_tam"
        data.to_csv(f"tweeteval-hashtags-gen/{dataset_name}/{split}.csv", index=False)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=["emoji", "emotion", "hate", "irony", "offensive", "sentiment", "stance"])
    parser.add_argument('--model', type=str, default='kp-times', choices=["bart-base", "bart-large", "kp-times"])
    parser.add_argument('--logging', action='store_true')
    parser.add_argument('--model_path', type=str, required=True)

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