from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
import json



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path'])


####################function for deployment
def copy_model_to_production():
    #copy the latest pickle file, the latestscore.txt value, and the ingestedfiles.txt file into the deployment directory
    cwd = os.getcwd()
    
    # Copy the trained model to prod_deployment_path
    os.popen(f'cp "{os.path.join(cwd, model_path, "trainedmodel.pkl")}" ' +
             f'"{os.path.join(cwd, prod_deployment_path, "trainedmodel.pkl")}"')

    # Copy the model score to prod_deployment_path
    os.popen(f'cp "{os.path.join(cwd, model_path, "latestscore.txt")}" ' +
             f'"{os.path.join(cwd, prod_deployment_path, "latestscore.txt")}"')

    # Copy a record of ingested data to prod_deployment_path
    os.popen(f'cp "{os.path.join(cwd, dataset_csv_path, "ingestedfiles.txt")}" ' +
             f'"{os.path.join(cwd, prod_deployment_path, "ingestedfiles.txt")}"')

if __name__ == '__main__':
    copy_model_to_production()
