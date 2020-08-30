#run notebook from requirments.txt venv
#allows us to plot graphs in notebook
import argparse
description = """   This script assumes to be run at 'evaluation' directory, have the retrained model
                    weights 'retrain_lodbrok.py' if not run the retrain script in model directory.
                    The script is made to show the results and performance of origninal model retrained
                    model against the test data set 'spambrainz_dataset_predict'. The output generated is
                    a confuison matrix which tells how many of the predicted values by both models match
                    the labels of the test_dataset.
                """

parser = argparse.ArgumentParser(description=description)
args = parser.parse_args()

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys
sys.path.append("..")
from numpy import loadtxt
from tensorflow.keras.models import load_model
import keras
from utils.evaluation import evaluate, print_stats


# function to plot confusion matrix

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):


    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def create_confusion_matrix(dataset, model, title):
    test_labels = dataset[:,0]
    converted_test_labels = []

    # labels are collected
    for label in test_labels:
        converted_test_labels.append(int(label))


    predict_data = {
        "main_input": np.array(dataset[:,1:10]),#.reshape(1,9),
        "email_input": np.array(dataset[:,10]),
        "website_input": np.array(dataset[:,11]),#.reshape(1),
        "bio_input": np.array(dataset[:,12:]),
    }

    #prediction done by model
    predictions = model.predict(x = [
        predict_data["main_input"],
        predict_data["email_input"],
        predict_data["website_input"],
        predict_data["bio_input"],
        ])

    rounded_predictions = []

    # integer predictions labels are collected
    for prediction in predictions:
        if(prediction[1]>prediction[0]):
            rounded_predictions.append(1)
        else:
            rounded_predictions.append(0)

    # initialize confusion matrix with test and predicted labels
    cm = confusion_matrix(converted_test_labels, rounded_predictions)

    # labels for cm
    cm_plot_labels = ['Non Spam Account','Spam Account']

    # plots the confusion matrix
    plot_confusion_matrix(cm, cm_plot_labels, title=title)


# gather editor account labels and prediction dataset
with open("../data/spambrainz_dataset_predict.pickle", "rb") as f:
        dataset = pickle.load(f)

# load original model
model = load_model('../models/weights/lodbrok1.h5')

title = "Confusion matrix for LodBrok"

# create original model confusion matrix
create_confusion_matrix(dataset, model, title)

#load retrained model
retrained_model = load_model('../models/weights/retrain_lodbrok.h5')

title = "Confusion matrix for retrained LodBrok"

# create retained model confusion matrix
create_confusion_matrix(dataset, retrained_model,title)