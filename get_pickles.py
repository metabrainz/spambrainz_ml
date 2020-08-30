import argparse
import requests

description = """  This script is used to download all the tokenizer/dataset pickle files needed to train, evaluate and predict using the model.
                   Also downloads model tokenizers used for preprocessing the datasets before feeding them to the model.
                   The script assumes to be run at spambrainz_ml top directory.
                   The files if present are overwritten by the script, kindly be careful."""

parser = argparse.ArgumentParser(description=description)

args = parser.parse_args()

# Tokenizer files reqeuried for preprocessing.py file
bio_tokenizer_url = 'https://github.com/metabrainz/spambrainz_ml/releases/download/v-0.1/bio_tokenizer.pickle'
web_tokenizer_url = 'https://github.com/metabrainz/spambrainz_ml/releases/download/v-0.1/web_tokenizer.pickle'
email_tokenizer_url = 'https://github.com/metabrainz/spambrainz_ml/releases/download/v-0.1/email_tokenizer.pickle'


bio_tokenizer_request = requests.get(bio_tokenizer_url, allow_redirects=True)
open('data/bio_tokenizer.pickle', 'wb').write(bio_tokenizer_request.content)

web_tokenizer_request = requests.get(web_tokenizer_url, allow_redirects=True)
open('data/web_tokenizer.pickle', 'wb').write(bio_tokenizer_request.content)

email_tokenizer_request = requests.get(email_tokenizer_url, allow_redirects=True)
open('data/email_tokenizer.pickle', 'wb').write(bio_tokenizer_request.content)

# Dataset files to for model training, evaluation and predictions

spambrainz_dataset_url = 'https://github.com/metabrainz/spambrainz_ml/releases/download/v-0.1/spambrainz_dataset.pickle'
spambrainz_dataset_eval_url = 'https://github.com/metabrainz/spambrainz_ml/releases/download/v-0.1/spambrainz_dataset_eval.pickle'
spambrainz_dataset_predict_url = 'https://github.com/metabrainz/spambrainz_ml/releases/download/v-0.1/spambrainz_dataset_predict.pickle'

spambrainz_dataset_request = requests.get(spambrainz_dataset_url, allow_redirects=True)
open('data/spambrainz_dataset.pickle', 'wb').write(spambrainz_dataset_request.content)

spambrainz_dataset_eval_request = requests.get(spambrainz_dataset_eval_url, allow_redirects=True)
open('data/spambrainz_dataset_eval.pickle', 'wb').write(spambrainz_dataset_eval_request.content)

spambrainz_dataset_predict_request = requests.get(spambrainz_dataset_predict_url, allow_redirects=True)
open('data/spambrainz_dataset_predict.pickle', 'wb').write(spambrainz_dataset_predict_request.content)
