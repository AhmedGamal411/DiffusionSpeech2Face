import torch
import pickle
import configparser
import sqlite3 as sl
import pandas as pd
import numpy as np

# TODO NOTEBOOK MAYBE

#text_embeds = torch.randn(4, 256, 768).cuda()
#print(text_embeds.shape)

# Loading configurations
configParser = configparser.RawConfigParser()   
configFilePath = r'configuration.txt'
configParser.read(configFilePath)
datasetPathDatabase =  configParser.get('COMMON', 'datasetPathDatabase') + '/dataset.db'

con = sl.connect(datasetPathDatabase)  # Connection to databases

# CREATE A VIEW
data = con.execute("SELECT V.ID, V.VIDEO_PATH, V.AGE, V.ETHNICITY, V.GENDER, A.SPEAKER_EMB, A.LANG  FROM VIDEO V INNER JOIN AUDIO A WHERE V.ID = A.VIDEO_ID")


#print(data.fetchall())
dataGotten = data.fetchall()
#print(dataGotten)

pd.set_option('display.max_columns', None)
df = pd.DataFrame(dataGotten,columns = ['ID','VIDEO_PATH','AGE','ETHNICITY','GENDER','SPEAKER_EMB','LANG'])
#print(df['SPEAKER_EMB'])

# unpickle speaker embedding tensor
df['SPEAKER_EMB'] = df['SPEAKER_EMB'].apply(lambda x:pickle.loads(x))
print(df.head(1)[''])