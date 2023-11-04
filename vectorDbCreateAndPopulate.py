import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
videos_per_user =  int(configParser.get('evaluate_imagen', 'videos_per_user'))
datasetPathDatabase =  configParser.get('COMMON', 'test_datasetPathDatabase') + '/dataset.db'

import pandas as pd
import lancedb
uri = datasetPathDatabase
db = lancedb.connect(uri)

try:
  db.drop_table("video")
except:
  print("")


files_to_insert = []
root_files = []
subroot_files = []
user_count = {}


for root, dirs, files in sorted(os.walk(datasetPathVideo)):
    #tasks = [insertIntoDb(file,root) for file in files]
    # TODO split into chunks of say 4 files
    #await asyncio.wait(tasks)
    #await asyncio.gather(*tasks)
    
    for file in files:
      if(not((root) in subroot_files)):
          user = os.path.basename(os.path.dirname(root))

          if(user_count.get(user,0) < videos_per_user ):
            user_count[user] = user_count.get(user,0) + 1
            files_to_insert.append(root +'/'+ file)
            subroot_files.append(root)
            root_files.append(str(os.path.dirname(root)))
    
print(user_count)
#for root_file in root_files:
#  print(root_file)

import pandas as pd
import numpy as np
df = pd.DataFrame()
i = 0
for file in files_to_insert:
    p_small = pathlib.Path(os.path.dirname(file))
    p_big = p_small.parent.absolute()
    df = df.append({'id': i, 'video_path': file, 'face_path': '', 'user' : str(p_big.name),
        'blurred_face_path': '', 'features_path': ''
        ,'vector' : np.random.rand(1),'stage': ''}, ignore_index=True)
    i = i +1

db.create_table("video", df)

print('INSERTED ' + str(len(df)) + ' VIDEOS')