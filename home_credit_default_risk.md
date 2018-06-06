
# Kaggle Home Credit Default Risk Competition

https://www.kaggle.com/c/home-credit-default-risk

This competition is sponsored by [Home Credit](http://www.homecredit.net), whose mission is to provide a positive and safe borrowing experience to groups of people that traditional, mainstream banks and financial institutions typically refuse to serve.

Home Credit targets a demographic that typically has no recourse but to deal with shady characters such as loan sharks when borrowing money. Many of these unbanked individuals are hard-working, well-intentioned folks who, either due to circumstances beyond their control or past mistakes, have fallen through the financial system’s cracks.

Home Credit needs an algorithm that will take as inputs various financial and personal information originally taken from a loan applicant's profile, and then determine and output a probability of the applicant eventually repaying the loan. This probability will be in the range [0.0, 1.0], where 1.0 represents a 100% certainty that the applicant will repay the loan and 0.0 indicates that there is zero chance that the applicant will eventually repay. The algorithm will be tested on a set of 48,744 individuals who previously borrowed from Home Credit. A CSV file must be produced that contains one header row, and 48,744 prediction rows, where each prediction row contains both a user ID, the `SKI_ID_CURR` column, and the probability, the `TARGET` column, of that user repaying their loan. The file must be formatted as follows:

    SK_ID_CURR,TARGET
    100001,0.1
    100005,0.9
    100013,0.2
    etc.
    
Home Credit knows which borrowers ultimately paid off their loans, and which ones eventually defaulted. A good algorithm will need to predict a high probability of repayment for the majority of borrowers who did successfully repay their loans. This algorithm will also need to predict a low probability of repayment for the majority of borrowers who eventually defaulted on their loans.

## I. Data Exploration


```python
# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Pretty display for notebooks
%matplotlib inline
```

### Data Description

From https://www.kaggle.com/c/home-credit-default-risk/data:

1. **application_{train|test}.csv**
   * This is the main table, broken into two files for Train (with TARGET) and Test (without TARGET).
   * Static data for all applications. One row represents one loan in our data sample.
<p>
<p>
* **bureau.csv**
   * All client's previous credits provided by other financial institutions that were reported to Credit Bureau (for clients who have a loan in our sample).
   * For every loan in our sample, there are as many rows as number of credits the client had in Credit Bureau before the application date.
<p>
<p>
* **bureau_balance.csv**

  * Monthly balances of previous credits in Credit Bureau.
  * This table has one row for each month of history of every previous credit reported to Credit Bureau – i.e the table has (#loans in sample * # of relative previous credits * # of months where we have some history observable for the previous credits) rows.
<p>
<p>
* **previous_application.csv**

  * All previous applications for Home Credit loans of clients who have loans in our sample.
  * There is one row for each previous application related to loans in our data sample.
<p>
<p>
* **POS_CASH_balance.csv**

  * Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit.
  * This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credits * # of months in which we have some history observable for the previous credits) rows.
<p>
<p>
* **installments_payments.csv**

  * Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample.
  * There is a) one row for every payment that was made plus b) one row each for missed payment.
  * One row is equivalent to one payment of one installment OR one installment corresponding to one payment of one previous Home Credit credit related to loans in our sample.
<p>
<p>
* **credit_card_balance.csv**

  * Monthly balance snapshots of previous credit cards that the applicant has with Home Credit.
  * This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credit cards * # of months where we have some history observable for the previous credit card) rows.

![alt text](images/homecredit.png "Data Table Diagram")


```python
# Load the data tables
application_train_data = pd.read_csv("data/application_train.csv")
application_test_data = pd.read_csv("data/application_test.csv")
bureau_data = pd.read_csv("data/bureau.csv")
bureau_balance_data = pd.read_csv("data/bureau_balance.csv")
previous_application_data = pd.read_csv("data/previous_application.csv")
POS_CASH_balance_data = pd.read_csv("data/POS_CASH_balance.csv")
installments_payments_data = pd.read_csv("data/installments_payments.csv")
credit_card_balance_data = pd.read_csv("data/credit_card_balance.csv")
```

### 1. Main Data Table (application_{train|test}.csv)


```python
# Display the first five records
display(application_train_data.head(n=5))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>TARGET</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>CODE_GENDER</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>...</th>
      <th>FLAG_DOCUMENT_18</th>
      <th>FLAG_DOCUMENT_19</th>
      <th>FLAG_DOCUMENT_20</th>
      <th>FLAG_DOCUMENT_21</th>
      <th>AMT_REQ_CREDIT_BUREAU_HOUR</th>
      <th>AMT_REQ_CREDIT_BUREAU_DAY</th>
      <th>AMT_REQ_CREDIT_BUREAU_WEEK</th>
      <th>AMT_REQ_CREDIT_BUREAU_MON</th>
      <th>AMT_REQ_CREDIT_BUREAU_QRT</th>
      <th>AMT_REQ_CREDIT_BUREAU_YEAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100002</td>
      <td>1</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.0</td>
      <td>406597.5</td>
      <td>24700.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100003</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>270000.0</td>
      <td>1293502.5</td>
      <td>35698.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100004</td>
      <td>0</td>
      <td>Revolving loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>67500.0</td>
      <td>135000.0</td>
      <td>6750.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100006</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.0</td>
      <td>312682.5</td>
      <td>29686.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100007</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>121500.0</td>
      <td>513000.0</td>
      <td>21865.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 122 columns</p>
</div>



```python
# Total number of entries in training group
print("Total number of entries in training group: {}".format(application_train_data.shape[0]))
```

    Total number of entries in training group: 307511



```python
# Total number of entries in test group
print("Total number of entries in test group: {}".format(application_test_data.shape[0]))
```

    Total number of entries in test group: 48744


**Main Data Table Featureset Exploration**

1. **SK_ID_CURR**: ID of loan in our sample	
* **TARGET**: Target variable (1 - client with payment difficulties: he/she had late payment more than X days on at least one of the first Y installments of the loan in our sample, 0 - all other cases)	
* **NAME_CONTRACT_TYPE**: Identification if loan is cash or revolving	
* **CODE_GENDER**: Gender of the client	
* **FLAG_OWN_CAR**: Flag if the client owns a car	
* **FLAG_OWN_REALTY**: Flag if client owns a house or flat	
* **CNT_CHILDREN**: Number of children the client has	
* **AMT_INCOME_TOTAL**: Income of the client	
* **AMT_CREDIT**: Credit amount of the loan	
* **AMT_ANNUITY**: Loan annuity	
* **AMT_GOODS_PRICE**: For consumer loans it is the price of the goods for which the loan is given	
* **NAME_TYPE_SUITE**: Who was accompanying client when he was applying for the loan	
* **NAME_INCOME_TYPE**: Clients income type (businessman, working, maternity leave,Ö)	
* **NAME_EDUCATION_TYPE**: Level of highest education the client achieved
* **NAME_FAMILY_STATUS**: Family status of the client	
* **NAME_HOUSING_TYPE**: What is the housing situation of the client (renting, living with parents, ...)	
* **REGION_POPULATION_RELATIVE**: Normalized population of region where client lives (higher number means the client lives in more populated region) -- normalized
* **DAYS_BIRTH**: Client's age in days at the time of application -- time only relative to the application
* **DAYS_EMPLOYED**: How many days before the application the person started current employment -- time only relative to the application
* **DAYS_REGISTRATION**: How many days before the application did client change his registration -- time only relative to the application
* **DAYS_ID_PUBLISH**: How many days before the application did client change the identity document with which he applied for the loan -- time only relative to the application
* **OWN_CAR_AGE**: Age of client's car	
* **FLAG_MOBIL**: Did client provide mobile phone (1=YES, 0=NO)	
* **FLAG_EMP_PHONE**: Did client provide work phone (1=YES, 0=NO)	
* **FLAG_WORK_PHONE**: Did client provide home phone (1=YES, 0=NO)	
* **FLAG_CONT_MOBILE**: Was mobile phone reachable (1=YES, 0=NO)	
* **FLAG_PHONE**: Did client provide home phone (1=YES, 0=NO)	
* **FLAG_EMAIL**: Did client provide email (1=YES, 0=NO)	
* **OCCUPATION_TYPE**: What kind of occupation does the client have	
* **CNT_FAM_MEMBERS**: How many family members does client have	
* **REGION_RATING_CLIENT**: Our rating of the region where client lives (1,2,3)	
* **REGION_RATING_CLIENT_W_CITY**: Our rating of the region where client lives with taking city into account (1,2,3)	
* **WEEKDAY_APPR_PROCESS_START**: On which day of the week did the client apply for the loan	
* **HOUR_APPR_PROCESS_START**: Approximately at what hour did the client apply for the loan	rounded
* **REG_REGION_NOT_LIVE_REGION**: Flag if client's permanent address does not match contact address (1=different, 0=same, at region level)	
* **REG_REGION_NOT_WORK_REGION**: Flag if client's permanent address does not match work address (1=different, 0=same, at region level)	
* **LIVE_REGION_NOT_WORK_REGION**: Flag if client's contact address does not match work address (1=different, 0=same, at region level)	
* **REG_CITY_NOT_LIVE_CITY**: Flag if client's permanent address does not match contact address (1=different, 0=same, at city level)	
* **REG_CITY_NOT_WORK_CITY**: Flag if client's permanent address does not match work address (1=different, 0=same, at city level)	
* **LIVE_CITY_NOT_WORK_CITY**: Flag if client's contact address does not match work address (1=different, 0=same, at city level)	
* **ORGANIZATION_TYPE**: Type of organization where client works	
* **EXT_SOURCE_1**: Normalized score from external data source -- normalized
* **EXT_SOURCE_2**: Normalized score from external data source -- normalized
* **EXT_SOURCE_3**: Normalized score from external data source -- normalized
* **APARTMENTS_AVG**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **BASEMENTAREA_AVG**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **YEARS_BEGINEXPLUATATION_AVG**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **YEARS_BUILD_AVG**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **COMMONAREA_AVG**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **ELEVATORS_AVG**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **ENTRANCES_AVG**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **FLOORSMAX_AVG**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **FLOORSMIN_AVG**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **LANDAREA_AVG**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **LIVINGAPARTMENTS_AVG**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **LIVINGAREA_AVG**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **NONLIVINGAPARTMENTS_AVG**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **NONLIVINGAREA_AVG**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **APARTMENTS_MODE**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **BASEMENTAREA_MODE**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **YEARS_BEGINEXPLUATATION_MODE**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **YEARS_BUILD_MODE**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **COMMONAREA_MODE**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **ELEVATORS_MODE**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **ENTRANCES_MODE**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **FLOORSMAX_MODE**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **FLOORSMIN_MODE**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **LANDAREA_MODE**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **LIVINGAPARTMENTS_MODE**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **LIVINGAREA_MODE**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **NONLIVINGAPARTMENTS_MODE**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **NONLIVINGAREA_MODE**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **APARTMENTS_MEDI**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **BASEMENTAREA_MEDI**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **YEARS_BEGINEXPLUATATION_MEDI**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **YEARS_BUILD_MEDI**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **COMMONAREA_MEDI**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **ELEVATORS_MEDI**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **ENTRANCES_MEDI**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **FLOORSMAX_MEDI**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **FLOORSMIN_MEDI**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **LANDAREA_MEDI**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **LIVINGAPARTMENTS_MEDI**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **LIVINGAREA_MEDI**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **NONLIVINGAPARTMENTS_MEDI**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **NONLIVINGAREA_MEDI**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **FONDKAPREMONT_MODE**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **HOUSETYPE_MODE**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **TOTALAREA_MODE**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **WALLSMATERIAL_MODE**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **EMERGENCYSTATE_MODE**: Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor -- normalized
* **OBS_30_CNT_SOCIAL_CIRCLE**: How many observation of client's social surroundings with observable 30 DPD (days past due) default	
* **DEF_30_CNT_SOCIAL_CIRCLE**: How many observation of client's social surroundings defaulted on 30 DPD (days past due) 	
* **OBS_60_CNT_SOCIAL_CIRCLE**: How many observation of client's social surroundings with observable 60 DPD (days past due) default	
* **DEF_60_CNT_SOCIAL_CIRCLE**: How many observation of client's social surroundings defaulted on 60 (days past due) DPD	
* **DAYS_LAST_PHONE_CHANGE**: How many days before application did client change phone	
* **FLAG_DOCUMENT_2**: Did client provide document 2	
* **FLAG_DOCUMENT_3**: Did client provide document 3	
* **FLAG_DOCUMENT_4**: Did client provide document 4	
* **FLAG_DOCUMENT_5**: Did client provide document 5	
* **FLAG_DOCUMENT_6**: Did client provide document 6	
* **FLAG_DOCUMENT_7**: Did client provide document 7	
* **FLAG_DOCUMENT_8**: Did client provide document 8	
* **FLAG_DOCUMENT_9**: Did client provide document 9	
* **FLAG_DOCUMENT_10**: Did client provide document 10	
* **FLAG_DOCUMENT_11**: Did client provide document 11	
* **FLAG_DOCUMENT_12**: Did client provide document 12	
* **FLAG_DOCUMENT_13**: Did client provide document 13	
* **FLAG_DOCUMENT_14**: Did client provide document 14	
* **FLAG_DOCUMENT_15**: Did client provide document 15	
* **FLAG_DOCUMENT_16**: Did client provide document 16	
* **FLAG_DOCUMENT_17**: Did client provide document 17	
* **FLAG_DOCUMENT_18**: Did client provide document 18	
* **FLAG_DOCUMENT_19**: Did client provide document 19	
* **FLAG_DOCUMENT_20**: Did client provide document 20	
* **FLAG_DOCUMENT_21**: Did client provide document 21	
* **AMT_REQ_CREDIT_BUREAU_HOUR**: Number of enquiries to Credit Bureau about the client one hour before application
* **AMT_REQ_CREDIT_BUREAU_DAY**: Number of enquiries to Credit Bureau about the client one day before application (excluding one hour before application)	
* **AMT_REQ_CREDIT_BUREAU_WEEK**: Number of enquiries to Credit Bureau about the client one week before application (excluding one day before application)	
* **AMT_REQ_CREDIT_BUREAU_MON**: Number of enquiries to Credit Bureau about the client one month before application (excluding one week before application)	
* **AMT_REQ_CREDIT_BUREAU_QRT**: Number of enquiries to Credit Bureau about the client 3 month before application (excluding one month before application)	
* **AMT_REQ_CREDIT_BUREAU_YEAR**: Number of enquiries to Credit Bureau about the client one day year (excluding last 3 months before application)

### 2. Bureau Data Table (bureau_balance.csv)


```python
# Display the first five records
display(bureau_data.head(n=5))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>SK_ID_BUREAU</th>
      <th>CREDIT_ACTIVE</th>
      <th>CREDIT_CURRENCY</th>
      <th>DAYS_CREDIT</th>
      <th>CREDIT_DAY_OVERDUE</th>
      <th>DAYS_CREDIT_ENDDATE</th>
      <th>DAYS_ENDDATE_FACT</th>
      <th>AMT_CREDIT_MAX_OVERDUE</th>
      <th>CNT_CREDIT_PROLONG</th>
      <th>AMT_CREDIT_SUM</th>
      <th>AMT_CREDIT_SUM_DEBT</th>
      <th>AMT_CREDIT_SUM_LIMIT</th>
      <th>AMT_CREDIT_SUM_OVERDUE</th>
      <th>CREDIT_TYPE</th>
      <th>DAYS_CREDIT_UPDATE</th>
      <th>AMT_ANNUITY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>215354</td>
      <td>5714462</td>
      <td>Closed</td>
      <td>currency 1</td>
      <td>-497</td>
      <td>0</td>
      <td>-153.0</td>
      <td>-153.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>91323.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Consumer credit</td>
      <td>-131</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>215354</td>
      <td>5714463</td>
      <td>Active</td>
      <td>currency 1</td>
      <td>-208</td>
      <td>0</td>
      <td>1075.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>225000.0</td>
      <td>171342.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Credit card</td>
      <td>-20</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>215354</td>
      <td>5714464</td>
      <td>Active</td>
      <td>currency 1</td>
      <td>-203</td>
      <td>0</td>
      <td>528.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>464323.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Consumer credit</td>
      <td>-16</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>215354</td>
      <td>5714465</td>
      <td>Active</td>
      <td>currency 1</td>
      <td>-203</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>90000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Credit card</td>
      <td>-16</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>215354</td>
      <td>5714466</td>
      <td>Active</td>
      <td>currency 1</td>
      <td>-629</td>
      <td>0</td>
      <td>1197.0</td>
      <td>NaN</td>
      <td>77674.5</td>
      <td>0</td>
      <td>2700000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Consumer credit</td>
      <td>-21</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


**Bureau Data Table Featureset Exploration**

1. **SK_ID_CURR**: ID of loan in our sample - one loan in our sample can have 0,1,2 or more related previous credits in credit bureau -- hashed
* **SK_BUREAU_ID**: Recoded ID of previous Credit Bureau credit related to our loan (unique coding for each loan application) -- hashed
* **CREDIT_ACTIVE**: Status of the Credit Bureau (CB) reported credits	
* **CREDIT_CURRENCY**: Recoded currency of the Credit Bureau credit -- recoded
* **DAYS_CREDIT**: How many days before current application did client apply for Credit Bureau credit -- time only relative to the application
* **CREDIT_DAY_OVERDUE**: Number of days past due on CB credit at the time of application for related loan in our sample	
* **DAYS_CREDIT_ENDDATE**: Remaining duration of CB credit (in days) at the time of application in Home Credit -- time only relative to the application
* **DAYS_ENDDATE_FACT**: Days since CB credit ended at the time of application in Home Credit (only for closed credit) -- time only relative to the application
* **AMT_CREDIT_MAX_OVERDUE**: Maximal amount overdue on the Credit Bureau credit so far (at application date of loan in our sample)	
* **CNT_CREDIT_PROLONG**: How many times was the Credit Bureau credit prolonged	
* **AMT_CREDIT_SUM**: Current credit amount for the Credit Bureau credit
* **AMT_CREDIT_SUM_DEBT**: Current debt on Credit Bureau credit	
* **AMT_CREDIT_SUM_LIMIT**: Current credit limit of credit card reported in Credit Bureau	
* **AMT_CREDIT_SUM_OVERDUE**: Current amount overdue on Credit Bureau credit	
* **CREDIT_TYPE**: Type of Credit Bureau credit (Car, cash,...)	
* **DAYS_CREDIT_UPDATE**: How many days before loan application did last information about the Credit Bureau credit come -- time only relative to the application
* **AMT_ANNUITY**: Annuity of the Credit Bureau credit	

### 3. Bureau Balance Data Table (bureau_balance.csv)


```python
# Display the first five records
display(bureau_balance_data.head(n=5))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_BUREAU</th>
      <th>MONTHS_BALANCE</th>
      <th>STATUS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5715448</td>
      <td>0</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5715448</td>
      <td>-1</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5715448</td>
      <td>-2</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5715448</td>
      <td>-3</td>
      <td>C</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5715448</td>
      <td>-4</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>


**Bureau Balance Data Table Featureset Exploration**

1. **SK_BUREAU_ID**: Recoded ID of Credit Bureau credit (unique coding for each application) - use this to join to CREDIT_BUREAU table -- hashed
* **MONTHS_BALANCE**: Month of balance relative to application date (-1 means the freshest balance date) -- time only relative to the application
* **STATUS**: Status of Credit Bureau loan during the month (active, closed, DPD0-30,Ö [C means closed, X means status unknown, 0 means no DPD, 1 means maximal did during month between 1-30, 2 means DPD 31-60,Ö 5 means DPD 120+ or sold or written off ])

### 4. Previous Application Data Table (previous_application.csv)


```python
# Display the first five records
display(previous_application_data.head(n=5))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_PREV</th>
      <th>SK_ID_CURR</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>AMT_ANNUITY</th>
      <th>AMT_APPLICATION</th>
      <th>AMT_CREDIT</th>
      <th>AMT_DOWN_PAYMENT</th>
      <th>AMT_GOODS_PRICE</th>
      <th>WEEKDAY_APPR_PROCESS_START</th>
      <th>HOUR_APPR_PROCESS_START</th>
      <th>...</th>
      <th>NAME_SELLER_INDUSTRY</th>
      <th>CNT_PAYMENT</th>
      <th>NAME_YIELD_GROUP</th>
      <th>PRODUCT_COMBINATION</th>
      <th>DAYS_FIRST_DRAWING</th>
      <th>DAYS_FIRST_DUE</th>
      <th>DAYS_LAST_DUE_1ST_VERSION</th>
      <th>DAYS_LAST_DUE</th>
      <th>DAYS_TERMINATION</th>
      <th>NFLAG_INSURED_ON_APPROVAL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2030495</td>
      <td>271877</td>
      <td>Consumer loans</td>
      <td>1730.430</td>
      <td>17145.0</td>
      <td>17145.0</td>
      <td>0.0</td>
      <td>17145.0</td>
      <td>SATURDAY</td>
      <td>15</td>
      <td>...</td>
      <td>Connectivity</td>
      <td>12.0</td>
      <td>middle</td>
      <td>POS mobile with interest</td>
      <td>365243.0</td>
      <td>-42.0</td>
      <td>300.0</td>
      <td>-42.0</td>
      <td>-37.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2802425</td>
      <td>108129</td>
      <td>Cash loans</td>
      <td>25188.615</td>
      <td>607500.0</td>
      <td>679671.0</td>
      <td>NaN</td>
      <td>607500.0</td>
      <td>THURSDAY</td>
      <td>11</td>
      <td>...</td>
      <td>XNA</td>
      <td>36.0</td>
      <td>low_action</td>
      <td>Cash X-Sell: low</td>
      <td>365243.0</td>
      <td>-134.0</td>
      <td>916.0</td>
      <td>365243.0</td>
      <td>365243.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2523466</td>
      <td>122040</td>
      <td>Cash loans</td>
      <td>15060.735</td>
      <td>112500.0</td>
      <td>136444.5</td>
      <td>NaN</td>
      <td>112500.0</td>
      <td>TUESDAY</td>
      <td>11</td>
      <td>...</td>
      <td>XNA</td>
      <td>12.0</td>
      <td>high</td>
      <td>Cash X-Sell: high</td>
      <td>365243.0</td>
      <td>-271.0</td>
      <td>59.0</td>
      <td>365243.0</td>
      <td>365243.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2819243</td>
      <td>176158</td>
      <td>Cash loans</td>
      <td>47041.335</td>
      <td>450000.0</td>
      <td>470790.0</td>
      <td>NaN</td>
      <td>450000.0</td>
      <td>MONDAY</td>
      <td>7</td>
      <td>...</td>
      <td>XNA</td>
      <td>12.0</td>
      <td>middle</td>
      <td>Cash X-Sell: middle</td>
      <td>365243.0</td>
      <td>-482.0</td>
      <td>-152.0</td>
      <td>-182.0</td>
      <td>-177.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1784265</td>
      <td>202054</td>
      <td>Cash loans</td>
      <td>31924.395</td>
      <td>337500.0</td>
      <td>404055.0</td>
      <td>NaN</td>
      <td>337500.0</td>
      <td>THURSDAY</td>
      <td>9</td>
      <td>...</td>
      <td>XNA</td>
      <td>24.0</td>
      <td>high</td>
      <td>Cash Street: high</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 37 columns</p>
</div>


**Previous Application Data Table Featureset Exploration**

1. **SK_ID_PREV**: ID of previous credit in Home credit related to loan in our sample. (One loan in our sample can have 0,1,2 or more previous loan applications in Home Credit, previous application could, but not necessarily have to lead to credit) -- hashed
* **SK_ID_CURR**: ID of loan in our sample -- hashed
* **NAME_CONTRACT_TYPE**: Contract product type (Cash loan, consumer loan [POS] ,...) of the previous application	
* **AMT_ANNUITY**: Annuity of previous application	
* **AMT_APPLICATION**: For how much credit did client ask on the previous application	
* **AMT_CREDIT**: Final credit amount on the previous application. This differs from AMT_APPLICATION in a way that the AMT_APPLICATION is the amount for which the client initially applied for, but during our approval process he could have received different amount - AMT_CREDIT	
* **AMT_DOWN_PAYMENT**: Down payment on the previous application	
* **AMT_GOODS_PRICE**: Goods price of good that client asked for (if applicable) on the previous application	
* **WEEKDAY_APPR_PROCESS_START**: On which day of the week did the client apply for previous application	
* **HOUR_APPR_PROCESS_START**: Approximately at what day hour did the client apply for the previous application -- rounded
* **FLAG_LAST_APPL_PER_CONTRACT**: Flag if it was last application for the previous contract. Sometimes by mistake of client or our clerk there could be more applications for one single contract	
* **NFLAG_LAST_APPL_IN_DAY**: Flag if the application was the last application per day of the client. Sometimes clients apply for more applications a day. Rarely it could also be error in our system that one application is in the database twice	
* **RATE_DOWN_PAYMENT**: Down payment rate normalized on previous credit -- normalized
* **RATE_INTEREST_PRIMARY**: Interest rate normalized on previous credit -- normalized
* **RATE_INTEREST_PRIVILEGED**: Interest rate normalized on previous credit -- normalized
* **NAME_CASH_LOAN_PURPOSE**: Purpose of the cash loan	
* **NAME_CONTRACT_STATUS**: Contract status (approved, cancelled, ...) of previous application	
* **DAYS_DECISION**: Relative to current application when was the decision about previous application made	time only relative to the application
* **NAME_PAYMENT_TYPE**: Payment method that client chose to pay for the previous application	
* **CODE_REJECT_REASON**: Why was the previous application rejected	
* **NAME_TYPE_SUITE**: Who accompanied client when applying for the previous application	
* **NAME_CLIENT_TYPE**: Was the client old or new client when applying for the previous application	
* **NAME_GOODS_CATEGORY**: What kind of goods did the client apply for in the previous application	
* **NAME_PORTFOLIO**: Was the previous application for CASH, POS, CAR, Ö
* **NAME_PRODUCT_TYPE**: Was the previous application x-sell o walk-in	
* **CHANNEL_TYPE**: Through which channel we acquired the client on the previous application	
* **SELLERPLACE_AREA**: Selling area of seller place of the previous application	
* **NAME_SELLER_INDUSTRY**: The industry of the seller	
* **CNT_PAYMENT**: Term of previous credit at application of the previous application	
* **NAME_YIELD_GROUP**: Grouped interest rate into small medium and high of the previous application -- grouped
* **PRODUCT_COMBINATION**: Detailed product combination of the previous application	
* **DAYS_FIRST_DRAWING**: Relative to application date of current application when was the first disbursement of the previous application -- time only relative to the application
* **DAYS_FIRST_DUE**: Relative to application date of current application when was the first due supposed to be of the previous application -- time only relative to the application
* **DAYS_LAST_DUE_1ST_VERSION**: Relative to application date of current application when was the first due of the previous application -- time only relative to the application
* **DAYS_LAST_DUE**: Relative to application date of current application when was the last due date of the previous application -- time only relative to the application
* **DAYS_TERMINATION**: Relative to application date of current application when was the expected termination of the previous application -- time only relative to the application
* **NFLAG_INSURED_ON_APPROVAL**: Did the client requested insurance during the previous application

### 5. POS CASH Balance Data Table (POS_CASH_balance.csv)


```python
# Display the first five records
display(POS_CASH_balance_data.head(n=5))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_PREV</th>
      <th>SK_ID_CURR</th>
      <th>MONTHS_BALANCE</th>
      <th>CNT_INSTALMENT</th>
      <th>CNT_INSTALMENT_FUTURE</th>
      <th>NAME_CONTRACT_STATUS</th>
      <th>SK_DPD</th>
      <th>SK_DPD_DEF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1803195</td>
      <td>182943</td>
      <td>-31</td>
      <td>48.0</td>
      <td>45.0</td>
      <td>Active</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1715348</td>
      <td>367990</td>
      <td>-33</td>
      <td>36.0</td>
      <td>35.0</td>
      <td>Active</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1784872</td>
      <td>397406</td>
      <td>-32</td>
      <td>12.0</td>
      <td>9.0</td>
      <td>Active</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1903291</td>
      <td>269225</td>
      <td>-35</td>
      <td>48.0</td>
      <td>42.0</td>
      <td>Active</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2341044</td>
      <td>334279</td>
      <td>-35</td>
      <td>36.0</td>
      <td>35.0</td>
      <td>Active</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


**POS CASH Balance Data Table Featureset Exploration**

1. **SK_ID_PREV**: ID of previous credit in Home Credit related to loan in our sample. (One loan in our sample can have 0,1,2 or more previous loans in Home Credit)	
* **SK_ID_CURR**: ID of loan in our sample	
* **MONTHS_BALANCE**: Month of balance relative to application date (-1 means the information to the freshest monthly snapshot, 0 means the information at application - often it will be the same as -1 as many banks are not updating the information to Credit Bureau regularly ) -- time only relative to the application
* **CNT_INSTALMENT**: Term of previous credit (can change over time)	
* **CNT_INSTALMENT_FUTURE**: Installments left to pay on the previous credit	
* **NAME_CONTRACT_STATUS**: Contract status during the month	
* **SK_DPD**: DPD (days past due) during the month of previous credit	
* **SK_DPD_DEF**: DPD during the month with tolerance (debts with low loan amounts are ignored) of the previous credit

### 6. Installments Payments Data Table (installments_payments.csv)


```python
# Display the first five records
display(installments_payments_data.head(n=5))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_PREV</th>
      <th>SK_ID_CURR</th>
      <th>NUM_INSTALMENT_VERSION</th>
      <th>NUM_INSTALMENT_NUMBER</th>
      <th>DAYS_INSTALMENT</th>
      <th>DAYS_ENTRY_PAYMENT</th>
      <th>AMT_INSTALMENT</th>
      <th>AMT_PAYMENT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1054186</td>
      <td>161674</td>
      <td>1.0</td>
      <td>6</td>
      <td>-1180.0</td>
      <td>-1187.0</td>
      <td>6948.360</td>
      <td>6948.360</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1330831</td>
      <td>151639</td>
      <td>0.0</td>
      <td>34</td>
      <td>-2156.0</td>
      <td>-2156.0</td>
      <td>1716.525</td>
      <td>1716.525</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2085231</td>
      <td>193053</td>
      <td>2.0</td>
      <td>1</td>
      <td>-63.0</td>
      <td>-63.0</td>
      <td>25425.000</td>
      <td>25425.000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2452527</td>
      <td>199697</td>
      <td>1.0</td>
      <td>3</td>
      <td>-2418.0</td>
      <td>-2426.0</td>
      <td>24350.130</td>
      <td>24350.130</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2714724</td>
      <td>167756</td>
      <td>1.0</td>
      <td>2</td>
      <td>-1383.0</td>
      <td>-1366.0</td>
      <td>2165.040</td>
      <td>2160.585</td>
    </tr>
  </tbody>
</table>
</div>


**Installments Payments Data Table Featureset Exploration**

1. **SK_ID_PREV**: ID of previous credit in Home credit related to loan in our sample. (One loan in our sample can have 0,1,2 or more previous loans in Home Credit) -- hashed
* **SK_ID_CURR**: ID of loan in our sample -- hashed
* **NUM_INSTALMENT_VERSION**: Version of installment calendar (0 is for credit card) of previous credit. Change of installment version from month to month signifies that some parameter of payment calendar has changed	
* **NUM_INSTALMENT_NUMBER**: On which installment we observe payment	
* **DAYS_INSTALMENT**: When the installment of previous credit was supposed to be paid (relative to application date of current loan) -- time only relative to the application
* **DAYS_ENTRY_PAYMENT**: When was the installments of previous credit paid actually (relative to application date of current loan) -- time only relative to the application
* **AMT_INSTALMENT**: What was the prescribed installment amount of previous credit on this installment	
* **AMT_PAYMENT**: What the client actually paid on previous credit on this installment

### 7. Credit Card Balance Data Table (credit_card_balance.csv)


```python
# Display the first five records
display(credit_card_balance_data.head(n=5))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_PREV</th>
      <th>SK_ID_CURR</th>
      <th>MONTHS_BALANCE</th>
      <th>AMT_BALANCE</th>
      <th>AMT_CREDIT_LIMIT_ACTUAL</th>
      <th>AMT_DRAWINGS_ATM_CURRENT</th>
      <th>AMT_DRAWINGS_CURRENT</th>
      <th>AMT_DRAWINGS_OTHER_CURRENT</th>
      <th>AMT_DRAWINGS_POS_CURRENT</th>
      <th>AMT_INST_MIN_REGULARITY</th>
      <th>...</th>
      <th>AMT_RECIVABLE</th>
      <th>AMT_TOTAL_RECEIVABLE</th>
      <th>CNT_DRAWINGS_ATM_CURRENT</th>
      <th>CNT_DRAWINGS_CURRENT</th>
      <th>CNT_DRAWINGS_OTHER_CURRENT</th>
      <th>CNT_DRAWINGS_POS_CURRENT</th>
      <th>CNT_INSTALMENT_MATURE_CUM</th>
      <th>NAME_CONTRACT_STATUS</th>
      <th>SK_DPD</th>
      <th>SK_DPD_DEF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2562384</td>
      <td>378907</td>
      <td>-6</td>
      <td>56.970</td>
      <td>135000</td>
      <td>0.0</td>
      <td>877.5</td>
      <td>0.0</td>
      <td>877.5</td>
      <td>1700.325</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>35.0</td>
      <td>Active</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2582071</td>
      <td>363914</td>
      <td>-1</td>
      <td>63975.555</td>
      <td>45000</td>
      <td>2250.0</td>
      <td>2250.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2250.000</td>
      <td>...</td>
      <td>64875.555</td>
      <td>64875.555</td>
      <td>1.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>69.0</td>
      <td>Active</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1740877</td>
      <td>371185</td>
      <td>-7</td>
      <td>31815.225</td>
      <td>450000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2250.000</td>
      <td>...</td>
      <td>31460.085</td>
      <td>31460.085</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>30.0</td>
      <td>Active</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1389973</td>
      <td>337855</td>
      <td>-4</td>
      <td>236572.110</td>
      <td>225000</td>
      <td>2250.0</td>
      <td>2250.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>11795.760</td>
      <td>...</td>
      <td>233048.970</td>
      <td>233048.970</td>
      <td>1.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>Active</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1891521</td>
      <td>126868</td>
      <td>-1</td>
      <td>453919.455</td>
      <td>450000</td>
      <td>0.0</td>
      <td>11547.0</td>
      <td>0.0</td>
      <td>11547.0</td>
      <td>22924.890</td>
      <td>...</td>
      <td>453919.455</td>
      <td>453919.455</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>101.0</td>
      <td>Active</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>


**Credit Card Balance Data Table Featureset Exploration**

1. **SK_ID_PREV**: ID of previous credit in Home credit related to loan in our sample. (One loan in our sample can have 0,1,2 or more previous loans in Home Credit) -- hashed
* **SK_ID_CURR**: ID of loan in our sample -- hashed
* **MONTHS_BALANCE**: Month of balance relative to application date (-1 means the freshest balance date) -- time only relative to the application
* **AMT_BALANCE**: Balance during the month of previous credit	
* **AMT_CREDIT_LIMIT_ACTUAL**: Credit card limit during the month of the previous credit	
* **AMT_DRAWINGS_ATM_CURRENT**: Amount drawing at ATM during the month of the previous credit	
* **AMT_DRAWINGS_CURRENT**: Amount drawing during the month of the previous credit	
* **AMT_DRAWINGS_OTHER_CURRENT**: Amount of other drawings during the month of the previous credit	
* **AMT_DRAWINGS_POS_CURRENT**: Amount drawing or buying goods during the month of the previous credit	
* **AMT_INST_MIN_REGULARITY**: Minimal installment for this month of the previous credit	
* **AMT_PAYMENT_CURRENT**: How much did the client pay during the month on the previous credit	
* **AMT_PAYMENT_TOTAL_CURRENT**: How much did the client pay during the month in total on the previous credit	
* **AMT_RECEIVABLE_PRINCIPAL**: Amount receivable for principal on the previous credit	
* **AMT_RECIVABLE**: Amount receivable on the previous credit	
* **AMT_TOTAL_RECEIVABLE**: Total amount receivable on the previous credit
* **CNT_DRAWINGS_ATM_CURRENT**: Number of drawings at ATM during this month on the previous credit	
* **CNT_DRAWINGS_CURRENT**: Number of drawings during this month on the previous credit	
* **CNT_DRAWINGS_OTHER_CURRENT**: Number of other drawings during this month on the previous credit	
* **CNT_DRAWINGS_POS_CURRENT**: Number of drawings for goods during this month on the previous credit	
* **CNT_INSTALMENT_MATURE_CUM**: Number of paid installments on the previous credit	
* **NAME_CONTRACT_STATUS**: Contract status (active signed,...) on the previous credit	
* **SK_DPD**: DPD (Days past due) during the month on the previous credit
* **SK_DPD_DEF**: DPD (Days past due) during the month with tolerance (debts with low loan amounts are ignored) of the previous credit
