import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])


#################Function for model scoring
def score_model(data_path=test_data_path, model_path=output_model_path ):
    # this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    # it should write the result to the latestscore.txt file

    # Read in test data
    cwd = os.getcwd()
    df = pd.DataFrame(columns=['corporation',
                               'lastmonth_activity',
                               'lastyear_activity',
                               'number_of_employees',
                               'exited'],)
    # Consider the case where there is more than one data file
    for file in os.listdir(os.path.join(cwd, data_path)):
        if file.split('.')[-1] == 'csv':
            df = df.append(
                pd.read_csv(os.path.join(cwd, data_path, file), 
                encoding='utf-8'),
                ignore_index=True,
    )

    y = df['exited'].values.reshape(-1, 1).ravel().astype(np.int64)
    X = df.drop(['corporation', 'exited'], axis=1,).values.reshape(-1, 3)

    # Read in trained model
    model_file = os.path.join(cwd, model_path, 'trainedmodel.pkl')
    with open(model_file, 'rb') as f:
        clf = pickle.load(f)

    # Predict on test data
    preds = clf.predict(X)

    # Calculate F1-Score
    f1_score = metrics.f1_score(y, preds)

    # Save F1-score
    with open(os.path.join(cwd, model_path, 'latestscore.txt'), 'w') as f:
        f.write(f'{f1_score}\n')

    return f1_score


if __name__ == '__main__':
    score_model()