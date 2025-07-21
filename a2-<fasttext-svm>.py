# %%
import pandas as pd
import spacy
import emoji

nlp = spacy.load("en_core_web_sm")
df = pd.read_csv("nlp2-a2.train", sep="\t", skiprows=1, names=["id", "gender", "text"])

def is_emoji_token(token):
    return any(char in emoji.EMOJI_DATA for char in token.text)

def tokenize(text):
    return [token.text.lower()
            for token in nlp(text)
            if not token.is_space and (token.is_alpha or token.is_punct or is_emoji_token(token))]


df["tokens"] = df["text"].apply(tokenize)

print(df[["text", "tokens"]].head())


# %%
import numpy as np
from gensim.models.fasttext import load_facebook_vectors

print("Loading FastText model...")
model = load_facebook_vectors("/Users/guo/Downloads/cc.en.300.bin")  

def tweet_to_vec(tokens, model):
    vectors = [model[word] for word in tokens if word in model]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size) 

print("Vectorizing tweets...")
X_train = np.vstack(df["tokens"].apply(lambda tokens: tweet_to_vec(tokens, model)).values)

y_train = df["gender"].values

# %%
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC

clf = LinearSVC()

scores = cross_val_score(clf, X_train, y_train, cv=5)

print("Cross-validation scores:", scores)
print("Average accuracy:", scores.mean())


# %%
from sklearn.svm import LinearSVC

clf = LinearSVC()

clf.fit(X_train, y_train)


# %%
df_test = pd.read_csv("nlp2-a2.test", sep="\t", skiprows=1, names=["id", "gender", "text"])
df_test["tokens"] = df_test["text"].apply(tokenize)

X_test = np.vstack(df_test["tokens"].apply(lambda tokens: tweet_to_vec(tokens, model)).values)

y_pred = clf.predict(X_test)

output = pd.DataFrame({
    "id": df_test["id"],
    "gender": y_pred
})

output.to_csv("a2-fasttext-svm.predictions", sep="\t", index=False)



