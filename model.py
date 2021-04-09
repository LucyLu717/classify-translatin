from sklearn.linear_model import LogisticRegression
from preprocess import preprocess
from sklearn.metrics import f1_score
import argparse
from nltk.translate.bleu_score import sentence_bleu
import numpy as np


def logisticReg():
    xTr, yTr = preprocess("data/train.txt")
    xTe, yTe = preprocess("data/test.txt")

    xTr = [[float(ex[-1])] for ex in xTr]
    xTe = [[float(ex[-1])] for ex in xTe]

    clf = LogisticRegression(random_state=0).fit(xTr, yTr)
    yPred = clf.predict(xTe)
    print("Precision score: {0}".format(round(clf.score(xTe, yTe), 3)))
    print("F1 score: {0}".format(round(f1_score(yTe, yPred), 3)))


def compute_bleu(data, weights):
    scores = []
    for ex in data:
        score = sentence_bleu([ex[1]], ex[2], weights=weights, auto_reweigh=True)
        scores.append([score])
    return scores


def bleu_val():
    xTr_raw, yTr_raw = preprocess("data/train.txt")
    xTe_raw, yTe = preprocess("data/test.txt")

    weight_cand = [
        [0.5, 0.5, 0, 0],
        [0.25, 0.25, 0.25, 0.25],
        [0.2, 0.3, 0.3, 0.2],
        [0, 0, 0.5, 0.5],
        [0.1, 0.1, 0.1, 0.7],
        [0, 0.15, 0.15, 0.7],
        [0, 0, 0, 1],
    ]
    preds = []

    # Validation to select the best weight
    for weight in weight_cand:
        xTr = compute_bleu(xTr_raw, weight)
        cutoff = int(len(xTr) * 0.8)
        xTr, xVal = xTr[:cutoff], xTr[cutoff:]
        yTr, yVal = yTr_raw[:cutoff], yTr_raw[cutoff:]

        clf = LogisticRegression(random_state=0).fit(xTr, yTr)
        yPred = round(clf.score(xVal, yVal), 3)
        preds.append(yPred)
        print(weight, yPred)

    max_w = np.argmax(preds)
    # train on the best weight
    xTr = compute_bleu(xTr_raw, weight_cand[max_w])
    xTe = compute_bleu(xTe_raw, weight_cand[max_w])
    clf = LogisticRegression(random_state=0).fit(xTr, yTr_raw)
    yPred = clf.predict(xTe)

    print("Precision score: {0}".format(round(clf.score(xTe, yTe), 3)))
    print("F1 score: {0}".format(round(f1_score(yTe, yPred), 3)))


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model", nargs="+", type=str, help="name of the model: logreg bleu")

args = parser.parse_args()
model = args.model

if "logreg" in model:
    print("Model chosen: {0}".format("Logistic Regression"))
    logisticReg()
if "bleu" in model:
    print("Model chosen: {0}".format("Recomputing bleu score with validation"))
    bleu_val()
