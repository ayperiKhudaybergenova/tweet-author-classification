# Twitter Gender Classification: NLP & Machine Learning

A Natural Language Processing (NLP) and Machine Learning project focused on text classification using tweets from public figures.

The project explores how linguistic patterns and text-based features can be used to build supervised machine learning models for author classification.

## Project Overview

Text classification is a fundamental NLP task that involves assigning predefined categories to textual data.

This project investigates the use of machine learning techniques to classify tweet authors based on their writing patterns and linguistic characteristics.

**Objective:** Build text classification models using feature extraction and supervised learning techniques.

## Technical Skills & Technologies

- **Programming:** Python
- **Natural Language Processing:** Text preprocessing, tokenization, feature extraction
- **Machine Learning:** Supervised learning, text classification
- **Text Representation:** Bag-of-Words, TF-IDF
- **Libraries:** Scikit-learn, Pandas, NumPy
- **Development:** Git, GitHub

## Dataset

The project uses a collection of tweets from public figures.

Each data instance contains:

- Unique instance identifier
- Classification target (gender label)
- Tweet text

The dataset is provided in tab-separated format and divided into training and test sets.

## Project Structure

```text
twitter-gender-classification/
│
├── baseline.py
├── a2-<method>.py
├── nlp2-a2.train
├── nlp2-a2.test
├── a2-<method>.predictions
├── README.md
└── .gitignore
```

## Machine Learning Approach

The project focuses on extracting meaningful features from raw text and applying supervised machine learning algorithms.

Potential approaches include:

- Text preprocessing and normalization
- Bag-of-Words representation
- TF-IDF feature extraction
- Logistic Regression
- Support Vector Machines (SVM)
- Naive Bayes classification
- Model comparison and optimization

## Evaluation

Classification models can be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Running the Project

Install the dependencies:

```bash
pip install pandas numpy scikit-learn
```

Run an individual model:

```bash
python a2-<method>.py
```

The script generates predictions in the required tab-separated format.

## Learning Outcomes

This project demonstrates practical experience with:

- Transforming raw text into numerical representations
- Extracting linguistic and textual features
- Building supervised NLP classification pipelines
- Applying machine learning to social media text
- Evaluating classification models
- Developing reproducible Python workflows

## Academic Context

Originally developed as part of a university assignment in Statistical Natural Language Processing (SNLP).

The repository has been organized as a portfolio project to demonstrate practical NLP and machine learning skills.

---

**Technologies:** Python · NLP · Text Classification · Scikit-learn · TF-IDF · Machine Learning
