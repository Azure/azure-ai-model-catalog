from huggingface_hub import HfApi

import pandas as pd

import re

 

LIST_OF_COLUMNS = ['modelId', 'downloads',

                   'lastModified', 'tags', 'pipeline_tag']

 

 

hf_api = HfApi()

# Get all the1 models in the list

models = hf_api.list_models(

    full=True, sort='lastModified', direction=-1)

# Unpack all values from the generator object

required_data = [i for i in models]

 

daata_dict = {}

# Loop through the list

for data in required_data:

    # Loop through all the column present in the list

    for key in data.__dict__.keys():

        if key in LIST_OF_COLUMNS:

            # Check the dictionary already contains a value for that particular column

            if daata_dict.get(key) is None:

                # If the column and its value is not present then insert column and an empty list pair to the dictionary

                daata_dict[key] = []

            # Get the value for that particular column

            values = daata_dict.get(key)

            if key == 'tags':

                # If its tag column extract value if it is nonne then bydefault return a list with string Empty

                values.append(data.__dict__.get(key, ["Empty"]))

            else:

                values.append(data.__dict__.get(key, "Empty"))

            daata_dict[key] = values
TASK_NAME = ['fill-mask', 'token-classification', 'question-answering',

             'summarization', 'text-generation', 'text-classification', 'translation']

STRING_TO_CHECK = 'transformers'

# Convert dictionary to the dataframe

df = pd.DataFrame(daata_dict)

# Find the data with the model which will be having trasnfomer tag

df = df[df.tags.apply(lambda x: STRING_TO_CHECK in x)]

# Retrive the data whose task is in the list

df = df[df['pipeline_tag'].isin(TASK_NAME)]

df['lastModified'] = pd.to_datetime(df['lastModified'], errors='coerce')
#converting lastModified to datetime
df['Day'] = [d.date() for d in df['lastModified']]
#creating another coloumn Day to sort las 7 days updated models
df['Day'] = pd.to_datetime(df['Day'], errors='coerce')
#converting Day to datetime
import datetime as dt
range_max = df['Day'].max()
range_min = range_max - dt.timedelta(days=7)

#take slice with final week of data
lastweek_updated = df[(df['Day'] >= range_min) & 
               (df['Day'] <= range_max)]
#filtering models updated last 7 days
lastweek_updated_downloads = lastweek_updated[(lastweek_updated["downloads"])>=10]
#please change the .csv file name for each run 
lastweek_updated_downloads.to_csv("lastweek_updated_models_min10_9_29.csv")
