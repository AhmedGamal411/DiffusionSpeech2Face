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
datasetPathDatabase =  configParser.get('COMMON', 'datasetPathDatabase') + '/datasetAdditionalFE.db'

if(recreateDb != 0):
    filename = uuid.uuid4().hex
    try:
        os.rename(datasetPathDatabase,configParser.get('COMMON', 'datasetPathDatabase') + '/' + str(uuid.uuid4().hex))
    except:
       pass
con = sl.connect(datasetPathDatabase)


if(recreateDb != 0):

    con.execute("""
        CREATE TABLE FACE_EMB (
            ID INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            VIDEO_ID INTEGER NOT NULL,
            FACE_ID INTEGER NOT NULL,
            FACE_EMB BLOB
        );
    """)


    con.execute("""
        CREATE INDEX FACE_EMB_VIDEO_ID_IDX ON FACE_EMB (VIDEO_ID);
        
    """)

    con.execute("""
        CREATE INDEX FACE_EMB_FACE_ID_IDX ON FACE_EMB (FACE_ID);
        
    """)




