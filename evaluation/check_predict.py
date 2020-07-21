#shows how to predict using the model and how it works
import numpy as np
import sys
sys.path.append("..")
from numpy import loadtxt
from tensorflow.keras.models import load_model
import keras
from utils.evaluation import evaluate, print_stats
import pickle

# load model
model = load_model('../models/weights/lodbrok1.h5')

# load dataset
with open("../data/spambrainz_dataset_predict.pickle", "rb") as f:
    dataset = pickle.load(f)

# organizing data 
predict_data = {
    "main_input": np.array(dataset[:,1:10]),#.reshape(1,9),
    "email_input": np.array(dataset[:,10]),
    "website_input": np.array(dataset[:,11]),#.reshape(1),
    "bio_input": np.array(dataset[:,12:]),
}

#prediction done by model.
guesses = model.predict(x = [
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

