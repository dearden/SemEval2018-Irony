from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.feature_selection import RFE
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
import pandas as pd


logging.basicConfig(level=logging.INFO)


# Some of the code from provided SemEval-2018 Task 3 Baseline.
# Extracts the tweets and labels from the given file
def make_corpus(file_name):
    labels = []
    corpus = []
    with open(file_name, 'rt') as data:
        for line in data:
            if not line.lower().startswith("tweet index"):  # ignore column names
                line = line.rstrip()
                label = int(line.split("\t")[1])
                tweet = line.split("\t")[2]
                labels.append(label)
                corpus.append(tweet)

    return corpus, labels


# ranks the features and writes them to a csv
def rank_features(svc, forest, names, data, target):
    svc_coefs = svc.coef_.ravel()                           # Get SVM weights
    # svc_coefs = abs(svc_coefs)                            # Uncomment if you don't care about sign
    forest_ranks = forest.feature_importances_.ravel()      # Get Random Forest Weights

    # Standardising the values between 0 and 1
    # svc_coefs = [(i-min(svc_coefs))/(max(svc_coefs)-min(svc_coefs)) for i in svc_coefs]
    forest_ranks = [(i-min(forest_ranks))/(max(forest_ranks)-min(forest_ranks)) for i in forest_ranks]

    # svc_rfe = rfe_ranking(svc, names, data, target)       # Recursive Feature Elimination
    # for_rfe = rfe_ranking(forest, names, data, target)

    # rows = np.array([[c, r, s, f] for c, r, s, f in zip(svc_coefs, forest_ranks, svc_rfe, for_rfe)])
    rows = np.array([[c, r] for c, r in zip(svc_coefs, forest_ranks)])
    data = pd.DataFrame(data=rows, index=names,
                        columns=["SVC Coefficients", "Random Forest Ranking"])

    # Sort the data by absolute value of svm weight
    sorted_data = data.assign(f=abs(data['SVC Coefficients']))\
        .sort_values(by='f', ascending=False)\
        .drop('f', axis=1)

    sorted_data.to_csv('classifier_ranks.csv', sep='\t')    # Write to csv

    data.sort_values(by='SVC Coefficients', ascending=False).plot.bar()     # Plot the features as bar chart
    plt.show()


# Calculates the rfe rankings of a feature set
def rfe_ranking(classifier, names, data, target):
    # rank all features, i.e continue the elimination until the last one
    rfe = RFE(classifier, n_features_to_select=1)
    rfe.fit(data, target)

    # return sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names))
    return rfe.ranking_


# Makes the feature table
def make_training_feature_table(path):
    # make feature model from dataset
    corpus, labels = make_corpus(path)

    iro_feats = IronyFeatures(corpus)
    feats = iro_feats.feature_table             # Get feature table
    feat_names = iro_feats.feat_names           # Get feature names
    return feats, feat_names, corpus, labels


# Code taken from SemEval 2018 Task 3 Baseline
if __name__ == "__main__":
    # Experiment Settings
    TASK = "A"
    DATA_PATH = "./SemEval2018-T4-train-task" + TASK + ".txt"
    FNAME = "./predictions-task" + TASK + ".txt"
    PREDICTIONS = open(FNAME, 'w')
    EVAL_PRED = open("evaluation/submission/res/predictions-task" + TASK + ".txt", "w")

    K_FOLDS = 10  # for cross-validation (number of folds)
    CLF = LinearSVC()  # default, non parameter-optimised linear kernel SVM (whatever that means)
    FOREST = RandomForestClassifier()  # Random Forest

    feats, feat_names, corpus, labels = make_training_feature_table(DATA_PATH)

    # creates array of predictions
    predicted = cross_val_predict(CLF, feats, labels, cv=K_FOLDS)           # use LinearSVC classifier
    # predicted = cross_val_predict(FOREST, feats, labels, cv=K_FOLDS)      # use random forest classifier

    # F1 Calculation (based on task)
    if TASK.lower() == 'a':
        score = metrics.f1_score(labels, predicted, pos_label=1)
        precision = metrics.average_precision_score(labels, predicted)
        recall = metrics.recall_score(labels, predicted)
        accuracy = metrics.accuracy_score(labels, predicted)

        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("Accuracy: " + str(accuracy))
    elif TASK.lower() == 'b':
        # More info by running evaluation script on results.
        score = metrics.f1_score(labels, predicted, average="macro")
    print("F1-Score Task", TASK, score)

    print('\n')
    print(metrics.confusion_matrix(labels, predicted))

    # Write predictions to evaluation file and local directory
    for p in predicted:
        PREDICTIONS.write("{}\n".format(p))
        EVAL_PRED.write("{}\n".format(p))
    PREDICTIONS.close()
    EVAL_PRED.close()

    # Fit classifiers for rankings
    CLF.fit(feats, labels)
    FOREST.fit(feats, labels)
    rank_features(CLF, FOREST, feat_names, feats, labels)

    # Write the features to a file.
    feat_table = np.concatenate(([feat_names], feats), axis=0)
    with open('feature_table.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerows(feat_table)

    # # prints out a breakdown of sentiment.
    # # Demonstrates the way sentiment transitions are determined.
    # for line in corpus:
    #     print_coloured_sentence(line)


