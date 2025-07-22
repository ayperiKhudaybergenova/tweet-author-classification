{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6eb49d64-f9e0-4b46-9901-f9821096aaf3",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "import pandas as pd\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "train_df = pd.read_csv('nlp2-a2.train', sep='\\t', header=None, names=['id', 'gender', 'text'])\n",
    "test_df = pd.read_csv('nlp2-a2.test', sep='\\t', header=None, names=['id', 'gender', 'text'])\n",
    "\n",
    "X_train = train_df['text']\n",
    "y_train = train_df['gender']\n",
    "X_test = test_df['text']\n",
    "test_ids = test_df['id']\n",
    "\n",
    "vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)\n",
    "X_train_vec = vectorizer.fit_transform(X_train)\n",
    "X_test_vec = vectorizer.transform(X_test)\n",
    "\n",
    "clf = LogisticRegression(max_iter=1000)\n",
    "clf.fit(X_train_vec, y_train)\n",
    "predictions = clf.predict(X_test_vec)\n",
    "\n",
    "output_df = pd.DataFrame({\n",
    "    'id': test_ids,\n",
    "    'gender': predictions\n",
    "})\n",
    "\n",
    "output_df.to_csv('ta2-<logreg>.predictions.tsv', sep='\\t', index=False, header=False)\n",
    "\n",
    "print(\"✅ Predictions saved to 'a2-<logreg>.predictions.tsv'\")\n",
    "\n",
    "preds = pd.read_csv('a2-<logreg>.predictions.tsv', sep='\\t', header=None)\n",
    "preds.head()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
