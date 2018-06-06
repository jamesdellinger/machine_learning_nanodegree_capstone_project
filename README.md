# Capstone Project
## Machine Learning Engineer Nanodegree
## Topic: Specialization

*Applying machine learning algorithms and techniques to solve the [Home Credit Default Risk competition](https://www.kaggle.com/c/home-credit-default-risk) on Kaggle.*

### My capstone project proposal is here:

https://github.com/jamesdellinger/machine_learning_nanodegree_capstone_project/blob/master/proposal.pdf

### My capstone project report is here:



### The code for my competition solution is here:

[ipython notebook](https://github.com/jamesdellinger/machine_learning_nanodegree_boston_housing_project/blob/master/boston_housing.ipynb)

[html version](https://github.com/jamesdellinger/machine_learning_nanodegree_boston_housing_project/blob/master/report.html)

### The competition's data tables are available here:

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

## Competition Overview
For my capstone project I will participate in the [Home Credit Default Risk competition](https://www.kaggle.com/c/home-credit-default-risk) on Kaggle. The goal of the competition is to be able to predict the likelihood that an applicant will repay a loan. The competition is sponsored by [Home Credit](http://www.homecredit.net), whose mission is to provide a positive and safe borrowing experience to groups of people that traditional, mainstream banks and financial institutions typically refuse to serve.

Home Credit targets a demographic that typically has no recourse but to deal with shady characters such as loan sharks when borrowing money. Many of these unbanked individuals are hard-working, well-intentioned folks who, either due to circumstances beyond their control or past mistakes, have fallen through the financial system’s cracks.

Home Credit needs an algorithm that will take as inputs various personal and alternative financial information originally taken from a loan applicant's profile, and then determine a probability of the applicant eventually repaying the loan. This probability will be in the range [0.0, 1.0], where 1.0 represents a 100\% certainty that the applicant will repay the loan and 0.0 indicates that there is zero chance that the applicant will eventually repay. The algorithm will be tested on a set of 48,744 individuals who previously borrowed from Home Credit. A CSV file must be produced that contains one header row, and 48,744 prediction rows, where each prediction row contains both a user ID, the `SKI_ID_CURR` column, and the probability, the `TARGET` column, of that user repaying their loan. The file must be formatted as follows:

```
SK_ID_CURR,TARGET
100001,0.1
100005,0.9
100013,0.2
etc.
```

Home Credit knows which borrowers ultimately paid off their loans, and which ones eventually defaulted. A good algorithm will need to predict a high probability of repayment for the majority of borrowers who did successfully repay their loans. This algorithm will also need to predict a low probability of repayment for the majority of borrowers who eventually defaulted on their loans.

The scoring metric for submissions is area under the [ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic). The best performing submissions are ranked on the competition's [leaderboard](https://www.kaggle.com/c/home-credit-default-risk/leaderboard).
