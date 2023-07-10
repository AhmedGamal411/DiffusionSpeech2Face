import asyncio
import sqlite3 as sl
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

datasetPathVideo =  configParser.get('COMMON', 'datasetPathVideo')
recreateDb =  int(configParser.get('dbCreateAndPopulate', 'recreateDb'))
datasetPathDatabase =  configParser.get('COMMON', 'datasetPathDatabase') + '/datasetAdditional.db'

if(recreateDb != 0):
    filename = uuid.uuid4().hex
    try:
        os.rename(datasetPathDatabase,configParser.get('COMMON', 'datasetPathDatabase') + '/' + str(uuid.uuid4().hex))
    except:
       pass
con = sl.connect(datasetPathDatabase)


if(recreateDb != 0):

    con.execute("""
        CREATE TABLE AUDIO (
            ID INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            VIDEO_ID INTEGER NOT NULL,
            AUDIO_LENGTH INTEGER,
            SPEAKER_EMB BLOB,
            LANG TEXT,
            AUDIO_EMB BLOB,
            AUDIO_EMB2 BLOB,
            WAV_TO_VEC BLOB,
            PYANNOTE_TITANET BLOB,
            AUDIO_FEATURES BLOB
        );
    """)


    con.execute("""
        CREATE INDEX AUDIO_VIDEO_ID_IDX ON AUDIO (VIDEO_ID);
        
    """)

    con.execute("""
        CREATE INDEX AUDIO_AUDIO_LENGTH_IDX ON AUDIO (AUDIO_LENGTH);
        
    """)

