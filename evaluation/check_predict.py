# Shows how to predict using the model and how it works
import argparse
description = """   This script assumes to be run at 'evaluation' directory and is made to see
                    the predictions done by lodbrok model against the 'spambrainz_dataset_predict.pickle'
                    dataset, the label mentioning whether the account is spam or not isn't considered while
                    loading the dataset. The predictions are classified such that, if prediction is closer
                    to 1 it is a spam account and if it is closer to 0 then it is a non_spam account.
                """
parser = argparse.ArgumentParser(description=description)
args = parser.parse_args()
import numpy as np
import sys
sys.path.append("..")
from numpy import loadtxt
from tensorflow.keras.models import load_model
import keras
from utils.evaluation import evaluate, print_stats
import pickle

# Load model
model = load_model('../models/weights/lodbrok1.h5')

# Load dataset
with open("../data/spambrainz_dataset_predict.pickle", "rb") as f:
    dataset = pickle.load(f)

# Organizing data
predict_data = {
    "main_input": np.array(dataset[:,1:10]),
    "email_input": np.array(dataset[:,10]),
    "website_input": np.array(dataset[:,11]),
    "bio_input": np.array(dataset[:,12:]),
}
#prediction done by model.
guesses = model.predict(
    x = [
        predict_data["main_input"],
        predict_data["email_input"],
        predict_data["website_input"],
        predict_data["bio_input"],
    ])
print("Predictions on test dataset by LodBrok Model :")
for guess in guesses:
    if(guess[1]>guess[0]):
        print("spam editor")
    else:
        print("non spam editor")