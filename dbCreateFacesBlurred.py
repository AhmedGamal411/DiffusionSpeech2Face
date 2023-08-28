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

recreateDb =  int(configParser.get('dbCreateAndPopulate', 'recreateDb'))
datasetPathDatabase =  configParser.get('COMMON', 'datasetPathDatabase') + '/datasetFacesBlurred.db'

if(recreateDb != 0):
    filename = uuid.uuid4().hex
    try:
        os.rename(datasetPathDatabase,configParser.get('COMMON', 'datasetPathDatabase') + '/' + str(uuid.uuid4().hex))
    except:
       pass
con = sl.connect(datasetPathDatabase)


if(recreateDb != 0):

    con.execute("""
        CREATE TABLE FACES_BLURRED (
            ID INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            VIDEO_ID INTEGER NOT NULL,
            FACE_ID INTEGER NOT NULL,
            BLURRED_FACE_EMB BLOB
        );
    """)


    con.execute("""
        CREATE INDEX FACES_BLURRED_VIDEO_ID_IDX ON FACES_BLURRED (VIDEO_ID);
        
    """)

    con.execute("""
        CREATE INDEX FACES_BLURRED_FACE_ID_IDX ON FACES_BLURRED (FACE_ID);
        
    """)

