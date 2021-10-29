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
    with mlflow.start_run(experiment_id=1) as run:

        logging.info('Split data')
        train, test = train_test_split(processed_data,random_state=42, test_size=0.33,stratify=df[binomial_target])
        
        train_claims_only = train.loc[train[gamma_target]>0]

        logging.info('Fit Binomial Model')
        #### BINOMIAL MODEL
        binomial_model = lm.LogisticRegression(solver='liblinear',max_iter=1000)
        binomial_model.fit(train[features],train[binomial_target])
        binomial_preds = binomial_model.predict(test[features])
  
        mlflow.log_metric("binomial_acc",metrics.accuracy_score(test[binomial_target],binomial_preds))
        mlflow.log_metric("binomial_precision",metrics.precision_score(test[binomial_target],binomial_preds))

        conf_matrix = metrics.confusion_matrix(test[binomial_target],binomial_preds)

        fig = sns.heatmap(
            pd.DataFrame(conf_matrix,index=['True','False'],columns=['True','False']),
            annot=True,fmt="d",cmap='Blues', cbar=False, annot_kws={"size": 16})
        fig.set_title('Confusion Matrix')
        fig.set_xlabel('Predicted')
        fig.set_ylabel('Actual')
        fig.figure.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')
        os.remove('confusion_matrix.png')


        logging.info('Fit Poisson Model')
        #### POISSON MODEL
        poisson_model = lm.PoissonRegressor(max_iter=10000)
        poisson_model.fit(train_claims_only[features],train_claims_only[poisson_target])
        poisson_preds = poisson_model.predict(test[features])
        mlflow.log_metric("poisson_r2_score",metrics.r2_score(test[poisson_target],poisson_preds))
        mlflow.log_metric("poisson_rmse",metrics.mean_squared_error(test[poisson_target],poisson_preds, squared=False))
        

        logging.info('Fit Gamma Model')
        #### GAMMA MODEL
        gamma_model = lm.GammaRegressor(max_iter=10000)
        gamma_model.fit(train_claims_only[features],train_claims_only[gamma_target])
        gamma_preds = gamma_model.predict(test[features])
        mlflow.log_metric("gamma_r2_score",metrics.r2_score(test[gamma_target],gamma_preds))
        mlflow.log_metric("gamma_rmse",metrics.mean_squared_error(test[gamma_target],gamma_preds, squared=False))

        logging.info('Save Models')
        #### Logging Models

        mlflow.sklearn.log_model(binomial_model,"binomial_model")
        mlflow.sklearn.log_model(poisson_model,"poisson_model")
        mlflow.sklearn.log_model(gamma_model,"gamma_model")
        

if __name__ == '__main__':
    main()
