import sys
sys.path.append("..")
from numpy import loadtxt
from tensorflow.keras.models import load_model
import keras
from utils.evaluation import evaluate, print_stats
# load model
model = load_model('lodbrok1.h5')
# summarize model.
model.summary()
eval = evaluate("lodbrok1.h5", "test.pickle")
print_stats(eval)
