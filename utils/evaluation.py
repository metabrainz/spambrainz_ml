
import pickle
import numpy as np
from models.lodbrok import load_model


def split_entries(entries):
    """Split dataset entry into four subarrays for the individual inputs."""
    return [entries[:, 1:10], entries[:, 10], entries[:, 11], entries[:, 12:]]


def evaluate(model_path, data_path):
    """Evaluate model"""
    model = load_model(model_path)

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    preds = model.predict(split_entries(data))

    # Shape (, (acc, t1, t2, confidence))
    stats = np.zeros((len(data), 4))

    for i in range(len(data)):
        y = int(data[i][0])

        if preds[i][y] > 0.5:
            # Accurate prediction
            stats[i][0] = 1
        else:
            if y == 0:
                # False hit (T1 error)
                stats[i][1] = 1
            else:
                # Miss (T2 error)
                stats[i][2] = 1

        # Confidence
        stats[i][3] = preds[i][y]

    return stats


def print_stats(eval_run):
    """Print statistics from evaluation run"""
    acc = np.average(eval_run[:, 0])
    t1 = np.average(eval_run[:, 1])
    t2 = np.average(eval_run[:, 2])
    conf = eval_run[:, 3]

    print("Accuracy: {:.3f}, T1 error: {:.3f}, T2 error: {:.3f}\n".format(acc, t1, t2))
    print("Confidence: Avg. {:.3f}, Min. {:.3f}, Max. {:.3f}, Standard dev. {:.3f}".format(
        np.average(conf), np.min(conf), np.max(conf), np.std(conf)))
