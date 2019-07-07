# Topic Categorization with Naive Bayes and Multinomial Logistic Regression
## Update: [Obtained highest accuracy!](https://www.kaggle.com/c/project-2-topic-categorization-cs529-2018/leaderboard)
## About
- This is python implementation of Naive Bayes and Multinomial Logistic Regression from scratch.
- The goal here is to predict which newsgroup a given document belongs to. Check 'Problem Description' for more info.
- Sparse matrices are heavily used since only a handful number of words out of complete vocabulary(of ~61k words) appear in a given document.

## How to run
This code was written and tested on python3.6. It might run on older versions as well. Below are the instructions to install the dependencies and run the code.
- `sudo apt install python3-pip`
- `sudo pip3 install pandas numpy sklearn`
- `sudo pip install sklearn`
- `python3 main.py`

## Problem Description
The task is to implement Naive Bayes and Logistic Regression for document classification. The 20 Newsgroups dataset is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups. This collection has become a popular dataset for experiments in text applications of machine learning techniques, such as text classification and text clustering.

[Here](https://www.kaggle.com/c/project-2-topic-categorization-cs529-2018/leaderboard) is the link of the competition on Kaggle.
