# HashTation
Source code for the paper "Hashtag-Guided Low-Resource Tweet Classification"

## Description
The data is in `data` and is based on the data from [TweetEval](https://github.com/cardiffnlp/tweeteval).

The file `hashtag_generation.py` trains (and saves) the hashtag generation models, while the file `hashtag_generation_inference.py` uses the saved models to generate the hashtags. Lastly, the file `classification.py` does the classification and evaluation for the TweetEval tasks.
