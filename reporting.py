import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

from sklearn import metrics
from diagnostics import model_predictions


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path']) 


##############Function for reporting
def report_confusion_matrix():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    
    # read data from "test_data_path"
    cwd = os.getcwd()
    df_test = pd.DataFrame(columns=['corporation',
                                    'lastmonth_activity',
                                    'lastyear_activity',
                                    'number_of_employees',
                                    'exited'],)
    # Consider the case where there is more than one test_data file
    for file in os.listdir(os.path.join(cwd, test_data_path)):
        df_test = df_test.append(
            pd.read_csv(os.path.join(cwd, test_data_path, file), 
            encoding='utf-8', low_memory=False),
            ignore_index=True,
        )
    # Get predictions for test data
    preds = model_predictions(df_test)
    
    # Get ground truth labels from test data
    y_test = df_test['exited'].values.reshape(-1, 1).ravel().astype(np.int64)
    
    # Create confusion matrix
    disp = metrics.ConfusionMatrixDisplay(
        metrics.confusion_matrix(y_test, preds))

    # Plot confusion matrix and save plot
    disp.plot()
    plt.savefig(os.path.join(model_path, 'confusionmatrix.png'))
    


if __name__ == '__main__':
    report_confusion_matrix()
