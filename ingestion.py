import pandas as pd
import os
import json

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

#############Function for data ingestion
def merge_multiple_dataframe():
    # check for datasets, compile them together, and write to an output file
    # create dataframe to compile all data
    df = pd.DataFrame(columns=['corporation',
                               'lastmonth_activity',
                               'lastyear_activity',
                               'number_of_employees',
                               'exited'],)
    cwd = os.getcwd()

    # create to record ingestion steps
    with open(os.path.join(cwd, output_folder_path, 'ingestedfiles.txt'), 
              'w',
              encoding='utf-8') as record:
        
        # process all datafiles in "input_folder_path" and update record
        for file in os.listdir(input_folder_path):
            df = df.append(
                pd.read_csv(os.path.join(cwd, input_folder_path, file), 
                encoding='utf-8'),
                ignore_index=True,
            )
            record.write(f'{file}\n')

        # remove duplicates
        df.drop_duplicates(inplace=True, ignore_index=True,)

        # writing the dataset to an output file
        df.to_csv(
            os.path.join(cwd, output_folder_path, 'finaldata.csv'),
            index=False,
        )


if __name__ == '__main__':
    merge_multiple_dataframe()
