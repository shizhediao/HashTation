# HashTation
Source code for the paper "Hashtag-Guided Low-Resource Tweet Classification"

## Installation
You can install all the dependencies by `source install.sh`.

## File Description
All the data we used in our paper is in `data` folder, which is based on the data from [TweetEval](https://github.com/cardiffnlp/tweeteval).

The file `hashtag_generation.py` trains (and saves) the hashtag generation models, while the file `hashtag_generation_inference.py` uses the saved models to generate the hashtags. 
Lastly, the file `classification.py` does the classification and evaluation for the TweetEval tasks.

## Training
Step 1: Train and save the hashtag generation models for your task first.

`python hashtag_generation.py --dataset [DATA_NAME]`


Step 2: Generate the hashtags for your task.

`python hashtag_generation_inference.py --dataset [DATA_NAME] --model_path [MODEL_PATH]`


Step 3: Train the classification model and evaluate it on your task.
`python classification.py`

where `DATA_NAME` could be `emoji, emotion, hate, irony, offensive, sentiment, stance, unified`.


## Citation
If you find this repository useful, you may cite [our paper]() as:  
```
@inproceedings{diao-etal-2023-hashtation,
    title={Hashtag-Guided Low-Resource Tweet Classification}, 
    author={Shizhe Diao and Sedrick Scott Keh and Liangming Pan and Zhiliang Tian and Yan Song and Tong Zhang},
    year={2023},
    booktitle={The Web Conference 2023}
}
```

## Contact
For help or issues using this package, please submit a GitHub issue.

For personal communication related to this package, please contact Shizhe Diao (sdiaoaa@connect.ust.hk) and Sedrick Scott Keh (skeh@cs.cmu.edu).