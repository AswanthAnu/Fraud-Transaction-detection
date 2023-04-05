# Fraudulent Transaction Detection: Project Overview
* Develop a model for predicting fraudulent transactions for a financial company and use insights from the model to develop an actionable plan.
* Data for the case is available in CSV format having 6362620 rows and 10 columns.
* Engineered features from the text and created more features.
* Optimized RandomForestClassifier and XGBClassifier used to reach the best model.
* Random Forest accuracy: 0.9999942371748326
* XG Boost accuracy: 0.9998643116619673

## Code and Resources Used
**Python Version:** 3.7

**Packages:** pandas, numpy, sklearn, xgboost, matplotlib, seaborn, warnings, flask, json, pickle.


**Dataset Link:** https://www.kaggle.com/datasets/chitwanmanchanda/fraudulent-transactions-data

## Features used
*	step
*	type
*	amount
*	oldbalanceOrg
*	newbalanceOrig 
*	oldbalanceDest
*	newbalanceDest 
*	errorBalanceOrg
*	errorBalanceDest
*	hoursOfDay 
*	isFraud

## Data Cleaning and EDA
After loading the data, I needed to clean it up so that it was usable for our model. And insight from the eda I made the following changes and created the following variables:

*	Created the three variables 'errorBalanceOrg', 'errorBalanceDest', 'hoursOfDay'.
*	Drop the three variables 'nameOrig', 'nameDest', 'isFlaggedFraud'.

Distributions of the data and the value counts for the various  variables. Below are a few highlights from the pivot tables.

![image](https://user-images.githubusercontent.com/101197982/230085295-19a9f8f5-9f83-49fd-9e8b-db190df0068b.png)
![image](https://user-images.githubusercontent.com/101197982/230085502-c13aaddf-69a0-49fe-86ad-2105b307266a.png)
![image](https://user-images.githubusercontent.com/101197982/230085644-8c25ea0f-d5db-4ddf-9663-f95c9ba9896d.png)

## Model Building 

First, I transformed the categorical variables into dummy variables. I also split the data into train and tests sets with a test size of 30%.   

Then do feature transformation on the variables to bring the values of the features to similar scale.

I tried two different models and evaluated them using Confustion matrix, accuracy_score and classification report.   

Models:
*	**RandomForestClassifier** – 'n_estimators':15, 'oob_score':True, 'class_weight':'balanced', 'n_jobs':-1, 'random_state':42.
*	**XGBClassifier** – 'max_depth':3, 'scale_pos_weight': weights, 'n_jobs':-1, 'random_state' : 42, 'learning_rate':0.1.

## Model performance
The Random Forest and XG Boost performed both performed well on the test and validation sets. 
*	**Random Forest** : 0.9999942371748326 accuracy score
![image](https://user-images.githubusercontent.com/101197982/230088598-95f90c19-484e-4e9b-88a4-427c654412dc.png)

*	**XG Boost**: 0.9998643116619673 accuracy score
![image](https://user-images.githubusercontent.com/101197982/230088734-1ae04e34-cbc0-42ac-b6c6-5e7373588de2.png)

In this fraud detection problem, two machine learning algorithms were employed - Random Forest and XG Boost. After evaluating their performance, it was found that Random Forest outperformed XG Boost.
