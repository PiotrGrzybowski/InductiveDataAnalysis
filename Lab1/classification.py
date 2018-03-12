import numpy as np
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
from sklearn.datasets import load_wine

from data.scripts.load_data import load_glass, load_diabets
import matplotlib.pyplot as plt
import pandas as pd

def accuracy(y_true, y_pred, average):
    return accuracy_score(y_true, y_pred)


def calculate_mean_and_deviation_from_scores(metrics, scores):
    return {metric: (np.mean(scores[metric]), np.var(scores[metric])) for metric in metrics}


def get_metrics_dict(metrics):
    return {metric: [] for metric in metrics}


def model_selection(data, target, model, metrics, splits):
    x, y = shuffle(data, target)
    scores = get_metrics_dict(metrics)

    for train_index, test_index in KFold(n_splits=splits).split(x):
        x_test, x_train, y_test, y_train = split_data_set(x, y, train_index, test_index)
        y_prediction = model.fit(x_train, y_train).predict(x_test)

        for metric in metrics:
            scores[metric].append(metric(y_test, y_prediction, average='macro'))

    return calculate_mean_and_deviation_from_scores(metrics, scores)


def binning_data_set():
    pass


def split_data_set(x, y, train_index, test_index):
    return x[test_index], x[train_index], y[test_index], y[train_index]


if __name__ == "__main__":
    metrics = [accuracy, f1_score, precision_score, recall_score]
    wine = load_wine()
    glass = load_glass()
    diabets = load_diabets()
    model = GaussianNB()

    data_set = glass
    scores = []
    folds = np.arange(2, 20)
    for i in folds:
        print(i)
        scores.append(model_selection(data_set['data'], data_set['target'], model, metrics, i))

    print(scores)

    for metric in metrics:
        means = [score[metric][0] for score in scores]
        deviations = [score[metric][1] for score in scores]
        plt.errorbar(folds, means, deviations, label=metric.__name__)

    plt.ylim((0, 0.6))
    plt.title("Metrics values depend on cross validation folds. Raw data.")
    plt.xlabel("Cross Validation Folds")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    df = pd.DataFrame(columns=['{}-Folds'.format(k) for k in folds],
                      index=['accuracy', 'f1_score', 'precission', 'recall'])

    for k in folds:
        df[str(k)] = pd.Series({'accuracy': make_value(result[accuracy]),
                                'f1_score': make_value(result[f1_score]),
                                'precission': make_value(result[precision_score]),
                                'recall': make_value(result[recall_score])})