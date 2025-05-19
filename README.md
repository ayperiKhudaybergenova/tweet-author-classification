# Assignment 2: Text classification

**Deadline:** Jul 25, 2025

In this assignment, 
we are interested in predicting the gender of an author from their tweets.
The data for this assignments comes from
a collabortive corpus collection effort from an earlier NLP class.
Each data instance is a (somewhat old) tweet from a public figure.

You can find the data in your repository as two separate files.
The file [nlp2-a2.train](nlp2-a2.train) contains the training instances,
and the file [nlp2-a2.test](nlp2-a2.test) contains the test instances.
Data is formatted as tab-separated files in both cases.
The columns include:

- column **1** the `id` of the instance (unique across train/test data)
- column **2** the outcome variable, `gender` of the author.
    The gender column in the test file is set to `_`
- column **3** the text

This time there are no numeric features.
You are required to extract features from the text yourself.
The provided baseline includes a simple example for simple linear classifiers.

Your task is to predict the gender of the authors in the test file,
and write your predictions to a file formatted as a tab-separated file
with two columns. The first column should be the id of the test instance
and the second should be the prediction. The following is an example
of the expected format.

```
id      gender
137671  male
12176   male
91952   female
47191   female
137759  female
81933   male
103410  female
...
```

The filename for the predictions file should be like `a2-<method>.predictions`,
where, `<method>` is a (short) string that identifies your approach.
The method name you use for the file will be visible on the scoreboard
(see below).
You are recommended to use understandable method names.
For example, if your method is based on a recurrent network,
`rnn`  may be a good choice.
You are encouraged to submit multiple sets of predictions
(e.g., one with a traditional method, one with a neural method).

For each approach, the main Python script should be named `a2-<method>.py`.
You are free to structure your code as multiple files
(and share code across different approaches).
However, the main file for each approach should be named as instructed above.

You are also provided with [a baseline](baseline.py) implementation of
a simple linear regression model using scikit-learn library. The example
also demonstrates the correct output format.
**Do not** modify the baseline,
even if you decide to improve on it, copy it to `a2-<method>.py`,
and modify the new file.

You are welcome to use any method, tool or library of your choice.
You can also make use of additional resources.

## Submissions and scoreboard

All assignments will be released through GitHub classroom.  The
assignment link will be provided on [the private course
repository](https://github.com/snlp2-2025/snlp2).

For each assignment we will keep a scoreboard
at [the private course repository](https://github.com/snlp2-2025/snlp2).
Each team is required to submit at least one prediction before the deadline.
You are welcome to submit new predictions as you improve your system
(the scoreboard will be updated once a day).
The scoreboard will show at most three (best) approaches from each team.

All code you use for the assignments should be pushed to your
assignment repository.
