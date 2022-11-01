import pickle
import diagnostics 
import scoring
import json
import os
import pandas as pd
import numpy as np
from flask import Flask, request


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path'])

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    # add return value for prediction outputs
    
    # Load data file
    data_file = request.files['file']
    data = pd.read_csv(data_file)

    # Read in trained model
    model_file = os.path.join(os.getcwd(), model_path, 'trainedmodel.pkl')
    with open(model_file, 'rb') as f:
        clf = pickle.load(f)

    # predict from data
    X = data.drop(['corporation', 'exited'], axis=1,).values.reshape(-1, 3)
    return str(clf.predict(X))

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    #check the score of the deployed model
    # add return value (a single F1 score number)
    return str(scoring.score_model())

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    # check means, medians, and modes for each column
    # return a list of all calculated summary statistics
    return str(diagnostics.dataframe_summary())

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnose():        
    # check timing and percent NA values
    # add return value for all diagnostics

    # run the timing, missing data, and dependency check functions
    return f'Execution time:\n{diagnostics.execution_time()}\n\n' + \
           f'Missing data:\n{diagnostics.missing_data()}\n\n' + \
           f'Dependencies:\n{diagnostics.outdated_packages_list()}\n\n'

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
