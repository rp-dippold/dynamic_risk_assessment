import os
import json
import pickle
import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting

import pandas as pd
import numpy as np

from sklearn import metrics

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = os.path.join(config['input_folder_path'])
output_folder_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

cwd = os.getcwd()

##################Check and read new data
#first, read ingestedfiles.txt
with open(os.path.join(cwd, prod_deployment_path, 'ingestedfiles.txt'), 'r') as f:
    ingested_files = [file.strip() for file in f.readlines()]

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
new_data = [
    file for file in os.listdir(os.path.join(cwd, input_folder_path))
    if file not in ingested_files]
print(new_data)
##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if len(new_data) > 0:
    # ingest data
    os.system('python3 ingestion.py')

    # read score from latest model
    with open(os.path.join(cwd, prod_deployment_path, 'latestscore.txt'), 'r') as f:
        latest_score = float(f.readline())

    # obtain score for new data
    new_score = float(scoring.score_model('ingesteddata', prod_deployment_path))
    
    ##################Checking for model drift
    #check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    if new_score < latest_score:

        ##################Deciding whether to proceed, part 2
        #if you found model drift, you should proceed. otherwise, do end the process here
        # train model with new data
        os.system('python3 training.py') 
        os.system('python3 scoring.py')  

        ##################Re-deployment
        #if you found evidence for model drift, re-run the deployment.py script
        os.system('python3 deployment.py')

        ##################Diagnostics and reporting
        #run diagnostics.py and reporting.py for the re-deployed model
        os.system('python3 reporting.py')
        os.system('python3 apicalls.py')
