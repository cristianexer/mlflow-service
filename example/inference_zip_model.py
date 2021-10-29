import os
import mlflow
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np
from dotenv import load_dotenv, find_dotenv
from sklearn.base import BaseEstimator
from sklearn import linear_model as lm
import seaborn as sns
import matplotlib.pyplot as plt
import logging


def main():

    logging.basicConfig(level=logging.INFO)

    # Load the environment variables
    load_dotenv(find_dotenv())

    # Load data
    logging.info("Loading data...")
    df = pd.read_csv('car_insurance_claim.csv')
    
    logging.info("Clean Data")
    # make columns lowercase
    df.columns = df.columns.str.lower()

    # drop useless columns
    df = df.drop(['kidsdriv','parent1','revoked','mvr_pts','travtime','id','birth'],axis=1)

    # clean money amounts
    df[['home_val','bluebook','oldclaim','clm_amt','income']] = df[['home_val','bluebook','oldclaim','clm_amt','income']].apply(lambda x: x.str.replace('$','').str.replace(',','')).astype(float)

    # clean values from columns
    to_clean = ['education','occupation','mstatus','gender','car_type']
    for col in to_clean:
        df[col] = df[col].str.replace('z_','').str.replace('<','')

    df['urbanicity'] = df['urbanicity'].str.split('/ ',expand=True)[1]

    to_clean = ['mstatus','red_car']
    for col in to_clean:
        df[col] = df[col].str.lower().replace({ 'yes': True, 'no': False}).astype(int)
        
    df = df.drop(['car_age','occupation','home_val','income','yoj'],axis=1).dropna()
    
    logging.info('Define features')
    features = ['age','gender','car_type','red_car','tif','education','car_use','bluebook','oldclaim','urbanicity']
    binomial_target = 'claim_flag'
    gamma_target = 'clm_amt'
    poisson_target = 'clm_freq'

    logging.info('Get Dummies')

    processed_data = pd.get_dummies(df[features + [binomial_target, gamma_target, poisson_target ]])
    features = list(processed_data.columns)

    logging.info('Start MLFlow Experiment')
    with mlflow.start_run(run_name="zip-model") as run:

        binomial_model = mlflow.sklearn.load_model("runs:/797d72b44b1b49b5a1297de75bc8373a/binomial_model")
        gamma_model = mlflow.sklearn.load_model("runs:/797d72b44b1b49b5a1297de75bc8373a/gamma_model")
        poisson_model = mlflow.sklearn.load_model("runs:/797d72b44b1b49b5a1297de75bc8373a/poisson_model")

        binomial_predictions = binomial_model.predict_proba(processed_data[features])[:,1]
        gamma_predictions = gamma_model.predict(processed_data[features])
        poisson_predictions = poisson_model.predict(processed_data[features])

        loss_cost = (binomial_predictions * poisson_predictions) * gamma_predictions

        print(loss_cost)
        

if __name__ == '__main__':
    main()
