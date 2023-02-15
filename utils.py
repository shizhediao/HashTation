import os
import sys
import logging
import datetime

def set_logger(log_to_file):
    if not os.path.exists('./logging'):
        os.makedirs('./logging')
    
    currtime = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(f"logging/{currtime}.log"),
            logging.StreamHandler()
        ] if log_to_file 
        else [logging.StreamHandler()]
    )

def process_results(results, dataset):
    # emoji, emotion, hate, offensive (Macro f1)
    if dataset in ['emoji', 'emotion', 'hate', 'offensive']:
        tweeteval_result = results['macro avg']['f1-score'] 

    # Irony (Irony class f1)
    elif dataset=='irony':
        tweeteval_result = results['1']['f1-score'] 

    # Sentiment (Macro Recall)
    elif dataset=='sentiment':
        tweeteval_result = results['macro avg']['recall']

    # Stance (Macro F1 of 'favor' and 'against' classes)
    elif dataset=='stance':
        f1_against = results['1']['f1-score']
        f1_favor = results['2']['f1-score']
        tweeteval_result = (f1_against+f1_favor) / 2 
    
    return tweeteval_result