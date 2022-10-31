import pandas as pd
import numpy as np
import timeit
import os
import pickle
import json
import subprocess


##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

##################Function to get model predictions
def model_predictions(data):
    #read the deployed model and a test dataset, calculate predictions
    cwd = os.getcwd()
    # read deployed model
    model_file = os.path.join(cwd, prod_deployment_path, 'trainedmodel.pkl')
    with open(model_file, 'rb') as f:
        clf = pickle.load(f)
    
    # prepare data for prediction
    X_test = data.drop(['corporation', 'exited'], axis=1,).values.reshape(-1, 3)

    # make predictions on data
    preds = clf.predict(X_test)

    #return value should be a list containing all predictions
    return preds

##################Function to get summary statistics
def dataframe_summary(data):
    # calculate summary statistics here
    # Calculate means, medians, and standard deviations for numerical columns.
    summary = []

    for col in data.columns:
        try:
            data[col] = pd.to_numeric(data[col])
            summary.append(
                (col,
                 np.mean(data[col]),
                 np.median(data[col]),
                 np.std(data[col]))
            )
        except ValueError:
            pass

    # return value should be a list containing all summary statistics
    return summary

##################Function to get missing values
def missing_data(data):
    # calculate the percentage of missing data in each column.
    summary = []
    for col in data.columns:
        summary.append(data[col].isna().sum() / data.shape[0] * 100)

    return summary

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    timings = []
    
    # obtain duration for ingestion.py
    starttime = timeit.default_timer()
    os.system('python3 ingestion.py')
    timings.append(timeit.default_timer() - starttime)

    # obtain duratoin of training.py
    starttime = timeit.default_timer()
    os.system('python3 training.py')
    timings.append(timeit.default_timer() - starttime)
    
    # return a list of 2 timing values in seconds
    return timings

##################Function to check dependencies
def outdated_packages_list(print_out=True):
    #get a list of outdated packages installed from requirements.txt

    # obtain all outdated packages
    # assumption: all packages of requirements.txt were installed before.
    outdated = subprocess.check_output(['pip', 'list','--outdated'])
    outdated_list = list()
    for i, line in enumerate(outdated.splitlines()):
        if i > 1:
            outdated_list.append(
                [elem.decode('utf-8') for elem in line.split()][0:3]
            )
    
    # print table
    if print_out:
        print(f'{"PACKAGE":<20}{"INSTALLED":<15}{"LATEST":<10}')
        print("---------------------------------------------")
        for row in outdated_list:
            print(f'{row[0]:<20}{row[1]:<15}{row[2]:<10}')

    return outdated_list


if __name__ == '__main__':
    # read test data from test_data_path
        # Read in test data
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
    preds = model_predictions(df_test)
    stat_summary = dataframe_summary(df_test)
    mv_summary = missing_data(df_test)
    exec_times = execution_time()
    outdated_pkgs = outdated_packages_list()
