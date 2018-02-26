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
from analysis import get_top_50_examples
from analysis import look_for_keywords

logging.basicConfig(level=logging.INFO)


# Extracts the tweets and labels from the given file
def make_test_corpus(file_name):
    corpus = []
    with open(file_name, 'rt') as data:
        for line in data:
            if not line.lower().startswith("tweet index"):  # ignore column names
                line = line.rstrip()
                tweet = line.split("\t")[1]
                corpus.append(tweet)
    return corpus


# Makes the feature table
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

    # fit the SVC classifier on training data and predict the labels for test data
    CLF.fit(train_feats, train_labels)
    predicted = CLF.predict(test_feats)

    # # fit the Random Forest classifier on training data and predict the labels for test data
    # FOREST.fit(train_feats, train_labels)
    # predicted = FOREST.predict(test_feats)

    # get the gold labels
    LABELS_PATH = "goldstandard_test_" + TASK + ".txt"
    with open(LABELS_PATH, 'r') as lab_file:
        labels = [int(x.strip()) for x in lab_file.readlines()]
    # print the confusion matrix and score
    print(metrics.confusion_matrix(labels, predicted))
    print(metrics.f1_score(labels, predicted, average="macro"))

    # Write the predictions to both the evaluation folder and the local directory
    for p in predicted:
        PREDICTIONS.write("{}\n".format(p))
        EVAL_PRED.write("{}\n".format(p))
    PREDICTIONS.close()
    EVAL_PRED.close()

    # Write the feature table to a file
    feat_table = np.concatenate(([feat_names], test_feats), axis=0)
    with open('TEST_FEATS.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerows(feat_table)

    # for line, lab, pred in zip(test_corpus, labels, predicted):
    #     print_coloured_sentence(line, str(pred) + '\t' + str(lab))

    # # Print top 50 highest value occurences of a feature
    # feat_index = feat_names.index('Num Dupe Chars')
    # feat_count = [x[feat_index] for x in test_feats]
    # get_top_50_examples(test_corpus, labels, predicted, feat_count)

    # # Print the sentiments of the tweets containing certain key words
    # sent_index = feat_names.index('Mean Sent')
    # mean_sent = [x[sent_index] for x in test_feats]
    # look_for_keywords(test_corpus, labels, predicted, mean_sent, ["love", "great", "fun", "thanks"])
