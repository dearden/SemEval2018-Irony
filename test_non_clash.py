from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.datasets import dump_svmlight_file
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import tree
import numpy as np
import logging
from irony_features import IronyFeatures
import matplotlib.pyplot as plt
from visualise_sentiment import print_coloured_sentence
import csv
from classify import make_training_feature_table

logging.basicConfig(level=logging.INFO)


def make_test_corpus(file_name):
    corpus = []
    with open(file_name, 'rt') as data:
        for line in data:
            if not line.lower().startswith("tweet index"):  # ignore column names
                line = line.rstrip()
                tweet = line.split("\t")[1]
                corpus.append(tweet)
    return corpus


def make_test_feature_table(train, test):
    iro_feats = IronyFeatures(train, test)
    feats = iro_feats.feature_table
    return feats


if __name__ == "__main__":
    TASK = 'B'
    TRAINING_PATH = "./SemEval2018-T4-train-task" + TASK + ".txt"
    TEST_PATH = "SemEval2018-T3_input_test_task" + TASK + ".txt"
    FNAME = "./predictions-task" + TASK + ".txt"
    PREDICTIONS = open(FNAME, 'w')
    EVAL_PRED = open("evaluation/submission/res/predictions-task" + TASK + ".txt", "w")

    # The classifiers
    CLF = LinearSVC()
    FOREST = RandomForestClassifier()

    # make training feature table and labels
    train_feats, feat_names, train_corpus, train_labels = make_training_feature_table(TRAINING_PATH)

    # Make the test corpus and feature table
    test_corpus = make_test_corpus(TEST_PATH)
    test_feats = make_test_feature_table(train_corpus, test_corpus)

    # Remove all the "Ironic-by-clash" rows from training set
    train_feats = [x for x, y in zip(train_feats, train_labels) if y != 1]
    train_labels = [x for x in train_labels if x != 1]
    train_labels = [1 if x else 0 for x in train_labels]

    # Get gold labels
    LABELS_PATH = "goldstandard_test_" + TASK + ".txt"
    with open(LABELS_PATH, 'r') as lab_file:
        labels = [int(x.strip()) for x in lab_file.readlines()]

    # Remove all the "Ironic-by-clash" rows from test set
    test_feats = [x for x, y in zip(test_feats, labels) if y != 1]
    labels = [x for x in labels if x != 1]
    labels = [1 if x else 0 for x in labels]

    # fit the SVC classifier on training data and predict the labels for test data
    CLF.fit(train_feats, train_labels)
    predicted = CLF.predict(test_feats)

    # # fit the Random Forest classifier on training data and predict the labels for test data
    # FOREST.fit(train_feats, train_labels)
    # predicted = FOREST.predict(test_feats)

    # Print confusion matrix and scores
    print(metrics.confusion_matrix(labels, predicted))
    score = metrics.f1_score(labels, predicted, average="macro")
    precision = metrics.average_precision_score(labels, predicted)
    recall = metrics.recall_score(labels, predicted)
    accuracy = metrics.accuracy_score(labels, predicted)
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("Accuracy: " + str(accuracy))
    print("F1-Score Task", TASK, score)
