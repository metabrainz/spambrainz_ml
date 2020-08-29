import sys
sys.path.append("..")
from numpy import loadtxt
from tensorflow.keras.models import load_model
import keras
from utils.evaluation import evaluate, print_stats
# load model
model = load_model('../models/weights/retrain_lodbrok.h5')
# summarize model
model.summary()
eval = evaluate("../models/weights/retrain_lodbrok.h5", "../data/retrain_eval_dataset.pickle")
print_stats(eval)
