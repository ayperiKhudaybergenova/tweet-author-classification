#!/usr/bin/env python3

import argparse
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support


ap = argparse.ArgumentParser()
ap.add_argument('train_file')
ap.add_argument('test_file')
args = ap.parse_args()


def read_data(filename):
    id_, gender, text = [], [], []
    with open(filename, 'r') as f:
        csvr = csv.reader(f,delimiter='\t')
        header = next(csvr)
        for row in csvr:
            id_.append(int(row[0]))
            gender.append(row[1])
            text.append(row[2])

    return id_, gender, text

_, trn_gender, trn_text = read_data(args.train_file)
tst_id, _, tst_text = read_data(args.test_file)

vec = CountVectorizer(ngram_range=(1,1), lowercase=True)
trn_x = vec.fit_transform(trn_text)

tst_x = vec.transform(tst_text)

m = LogisticRegression(solver='liblinear')
m.fit(trn_x, trn_gender)
trn_pred = m.predict(trn_x)
p, r, f, _ = precision_recall_fscore_support(trn_gender, trn_pred,
                                             average='macro')
print("Training scores:") 
print(f" Precision: {p:0.4f}")
print(f" Recall:    {r:0.4f}")
print(f" F1-score:  {f:0.4f}")

pred = m.predict(tst_x)

with open('a2-baseline.predictions', 'wt') as f:
    print("id\tage", file=f)
    for id_, pred in zip(tst_id, pred):
        print(f"{id_}\t{pred}", file=f)
