import pathlib
import os
import configparser
from random import randrange
import uuid



# TODO document how to set up the needed folders and paths in configuration.txt
# TODO document how to use recreateDb
# TODO indices
configParser = configparser.RawConfigParser()   
configFilePath = r'configuration.txt'
configParser.read(configFilePath)

datasetPathVideo =  configParser.get('COMMON', 'test_datasetPathVideo')
datasetPathDatabase =  configParser.get('COMMON', 'test_datasetPathDatabase') + '/dataset.db'


import lancedb
uri = datasetPathDatabase
db = lancedb.connect(uri)


files_to_insert = []


for root, dirs, files in os.walk(datasetPathVideo):
    #tasks = [insertIntoDb(file,root) for file in files]
    # TODO split into chunks of say 4 files
    #await asyncio.wait(tasks)
    #await asyncio.gather(*tasks)
    for file in files:
        files_to_insert.append(root +'/'+ file)
        print(root + file)



import pandas as pd
import numpy as np
df = pd.DataFrame()
i = 0
for file in files_to_insert:
    df = df.append({'id': i, 'video_path': file, 'face_path': '', 
        'blurred_face_path': '', 'features_path': '', 'user' : ''
        ,'vector' : np.random.rand(1),'stage': ''}, ignore_index=True)
    i = i +1

db.create_table("video", df)

print('INSERTED ' + str(len(df)) + ' VIDEOS')