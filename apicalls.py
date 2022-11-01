import requests
import os
import json

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1"
PORT = 8000

#Call each API endpoint and store the responses
with open('testdata/testdata.csv', 'r') as f:
    response1 = requests.post(f'{URL}:{PORT}/prediction',
                              files={'file': f}).content

response2 = requests.get(f'{URL}:{PORT}/scoring').content
response3 = requests.get(f'{URL}:{PORT}/summarystats').content
response4 = requests.get(f'{URL}:{PORT}/diagnostics').content

#combine all API responses

responses = {'PREDICTION': response1.decode('utf-8'),
             'SCORING': response2.decode('utf-8'),
             'SUMMARY-STATISTICS': response3.decode('utf-8'),
             'DIAGNOSTICS': response4.decode('utf-8')
             }

#write the responses to your workspace
#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path'])

with open(os.path.join(os.getcwd(), model_path, 'apireturns.txt'), 'w') as f:
    for key, value in responses.items():
        if key == 'PREDICTION':
            f.write(f'{key}: {value}\n\n')
        elif key == 'SCORING':
            f.write(f'{key}: {value}\n\n')
        elif key == 'SUMMARY-STATISTICS':
            f.write(f'{key}:\n')
            values = value.strip('[]').split('),')
            for val in values:
                f.write(f'{val.strip(" (")}\n')
            f.write('\n')
        elif key == 'DIAGNOSTICS':
            f.write(f'{key}:\n')
            f.write(value)
