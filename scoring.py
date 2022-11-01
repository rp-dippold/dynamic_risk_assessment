import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])


#################Function for model scoring
def score_model():
    # this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    # it should write the result to the latestscore.txt file

    # Read in test data
    cwd = os.getcwd()
    df = pd.DataFrame(columns=['corporation',
                               'lastmonth_activity',
                               'lastyear_activity',
                               'number_of_employees',
                               'exited'],)
    # Consider the case where there is more than one test_data file
    for file in os.listdir(os.path.join(cwd, test_data_path)):
        df = df.append(
            pd.read_csv(os.path.join(cwd, test_data_path, file), 
            encoding='utf-8'),
            ignore_index=True,
    )

    y_test = df['exited'].values.reshape(-1, 1).ravel().astype(np.int64)
    X_test = df.drop(['corporation', 'exited'], axis=1,).values.reshape(-1, 3)

    # Read in trained model
    model_file = os.path.join(cwd, model_path, 'trainedmodel.pkl')
    with open(model_file, 'rb') as f:
        clf = pickle.load(f)

    # Predict on test data
    preds = clf.predict(X_test)

    # Calculate F1-Score
    f1_score = metrics.f1_score(y_test, preds)

    # Save F1-score
    with open(os.path.join(cwd, model_path, 'latestscore.txt'), 'w') as f:
        f.write(f'{f1_score}\n')

    return f1_score


if __name__ == '__main__':
    score_model()