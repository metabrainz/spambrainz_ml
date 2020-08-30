import argparse


description = """   This script assumes to be run at 'evaluation' directory.
                    It requires 'lodbrok1.h5' and 'spambrainz_dataset_eval.pickle'
                    to run. The purpose of the script is to show the performance
                    of lodbrok model against evaluation dataset. The output is how
                    well the model performed over evaluation dataset"""

parser = argparse.ArgumentParser(description=description)

args = parser.parse_args()
import sys
sys.path.append("..")
from numpy import loadtxt
from tensorflow.keras.models import load_model
import keras
from utils.evaluation import evaluate, print_stats
# load model
model = load_model('../models/weights/lodbrok1.h5')
# summarize model
model.summary()
eval = evaluate("../models/weights/lodbrok1.h5", "../data/spambrainz_dataset_eval.pickle")
print_stats(eval)
