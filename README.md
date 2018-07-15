# Machine Learning Capstone Project
*Applying machine learning algorithms and techniques to solve the [Home Credit Default Risk competition](https://www.kaggle.com/c/home-credit-default-risk) on kaggle.*

<img src="https://github.com/jamesdellinger/machine_learning_nanodegree_capstone_project/blob/master/mlndlogo.png" height="140">

For Udacity's [Machine Learning Engineer](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t) Nanodegree.

Topic: Specialization.

## Overview
* I applied extensive exploration and preprocessing to the Home Credit Default Risk competition's [dataset](https://www.kaggle.com/c/home-credit-default-risk/data).
* I then fit a LightGBM classifier to this data in order predict which borrowers are most likely to have difficulty repaying their loans.

## My Capstone Project Report
* [Report pdf](https://github.com/jamesdellinger/machine_learning_nanodegree_capstone_project/blob/master/report.pdf)

## My Capstone Project Proposal
* [Proposal pdf](https://github.com/jamesdellinger/machine_learning_nanodegree_capstone_project/blob/master/proposal.pdf)

## My Competition Solution Code
* [ipython notebook](https://github.com/jamesdellinger/machine_learning_nanodegree_capstone_project/blob/master/home_credit_default_risk.ipynb) / [pdf version](https://github.com/jamesdellinger/machine_learning_nanodegree_capstone_project/blob/master/home_credit_default_risk.pdf)

## Project Grading and Evaluation
* [Capstone Project Review](https://github.com/jamesdellinger/machine_learning_nanodegree_capstone_project/blob/master/capstone_project_review.pdf)

* [Capstone Project Grading Rubric](https://github.com/jamesdellinger/machine_learning_nanodegree_capstone_project/blob/master/capstone_project_grading_rubric.pdf)

* [Capstone Project Proposal Review](https://github.com/jamesdellinger/machine_learning_nanodegree_capstone_project/blob/master/capstone_proposal_project_review.pdf)

* [Capstone Project Proposal Grading Rubric](https://github.com/jamesdellinger/machine_learning_nanodegree_capstone_project/blob/master/capstone_proposal_grading_rubric.pdf)

## Home Credit Default Risk Competition Synopsis
The goal of the [Home Credit Default Risk competition](https://www.kaggle.com/c/home-credit-default-risk) on Kaggle is the creation of a machine learning algorithm that is able to predict the likelihood that an loan applicant will make at least one late payment when repaying their loan. The competition is sponsored by [Home Credit](http://www.homecredit.net), whose mission is to provide a positive and safe borrowing experience to groups of people that traditional, mainstream banks and financial institutions typically refuse to serve.

Home Credit targets a demographic that typically has no recourse but to deal with shady characters such as loan sharks when borrowing money. Many of these unbanked individuals are hard-working, well-intentioned folks who, either due to circumstances beyond their control or past mistakes, have fallen through the financial system’s cracks.

Home Credit needs an algorithm that will take as inputs various personal and alternative financial information originally taken from a loan applicant's profile, and then determine a probability of the applicant eventually becoming delinquent. This probability will be in the range [0.0, 1.0], where 1.0 represents a 100\% certainty that the applicant will make at least one late payment and 0.0 indicates that there is zero chance that the applicant will ever be delinquent. The algorithm will be tested on a set of 48,744 individuals who previously borrowed from Home Credit. A CSV file must be produced that contains one header row, and 48,744 prediction rows, where each prediction row contains both a user ID, the `SKI_ID_CURR` column, and the probability, the `TARGET` column, of that user repaying their loan. The file must be formatted as follows:

```
SK_ID_CURR,TARGET
100001,0.1
100005,0.9
100013,0.2
etc.
```

Home Credit knows which borrowers ultimately made at least one late payment, and which ones were never delinquent. A good algorithm will need to predict a high probability of delinquency for the majority of borrowers who did make a late payment. This algorithm will also need to predict a low probability of delinquency for the majority of borrowers who always paid on time.

The scoring metric for submissions is area under the [ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic). The best performing submissions are ranked on the competition's [leaderboard](https://www.kaggle.com/c/home-credit-default-risk/leaderboard).

## Competition Data Tables
https://www.kaggle.com/c/home-credit-default-risk/data

In order to reproduce my results, the following eight CSV files must be downloaded and unzipped inside the `/data` directory:

1. [application_test.csv.zip](https://www.kaggle.com/c/9120/download/application_test.csv.zip)
2. [application_train.csv.zip](https://www.kaggle.com/c/9120/download/application_train.csv.zip)
3. [bureau.csv.zip](https://www.kaggle.com/c/9120/download/bureau.csv.zip)
4. [bureau_balance.csv.zip](https://www.kaggle.com/c/9120/download/bureau_balance.csv.zip)
5. [credit_card_balance.csv.zip](https://www.kaggle.com/c/9120/download/credit_card_balance.csv.zip)
6. [installments_payments.csv.zip](https://www.kaggle.com/c/9120/download/installments_payments.csv.zip)
7. [POS_CASH_balance.csv.zip](https://www.kaggle.com/c/9120/download/POS_CASH_balance.csv.zip)
8. [previous_application.csv.zip](https://www.kaggle.com/c/9120/download/previous_application.csv.zip)

## Dependencies
* [requirements.txt](https://github.com/jamesdellinger/machine_learning_nanodegree_capstone_project/blob/master/requirements.txt)

* [Anaconda .yml file](https://github.com/jamesdellinger/machine_learning_nanodegree_capstone_project/blob/master/home_credit_default_risk_competition.yml)
