import json
import pickle
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 


#################Function for training the model
def train_model():
    cwd = os.getcwd()
    # read in finaldata.csv
    df = pd.read_csv(os.path.join(cwd, dataset_csv_path, 'finaldata.csv'),
                     encoding='utf-8',)
    
    y_train = df['exited'].values.reshape(-1, 1).ravel().astype(np.int64)
    X_train = df.drop(['corporation', 'exited'], axis=1,).values.reshape(-1, 3)

    # use this logistic regression for training
    clf = LogisticRegression(C=1.0, class_weight=None, dual=False, 
                             fit_intercept=True, intercept_scaling=1, 
                             l1_ratio=None, max_iter=100, multi_class='auto', 
                             n_jobs=None, penalty='l2', random_state=0, 
                             solver='liblinear', tol=0.0001, verbose=0,
                             warm_start=False,)
    
    # fit the logistic regression to your data
    clf = clf.fit(X_train, y_train)    
    
    # write the trained model to your workspace in a file called trainedmodel.pkl
    file_name = os.path.join(cwd, model_path, 'trainedmodel.pkl')

    with open(file_name, 'wb') as fn:
        pickle.dump(clf, fn)


if __name__ == '__main__':
    train_model()
