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
datasetPathDatabase =  configParser.get('COMMON', 'datasetPathDatabase') + '/dataset.db'

if(recreateDb != 0):
    filename = uuid.uuid4().hex
    try:
        os.rename(datasetPathDatabase,configParser.get('COMMON', 'datasetPathDatabase') + '/' + str(uuid.uuid4().hex))
    except:
       pass
con = sl.connect(datasetPathDatabase)


if(recreateDb != 0):
    con.execute("""
        CREATE TABLE VIDEO (
            ID INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            VIDEO_PATH TEXT NOT NULL,
            AGE INTEGER,
            ETHNICITY INTEGER,
            GENDER INTEGER,
            AUDIO_PRE INTEGER,
            FACES_PRE INTEGER
        );
    """)

    con.execute("""
        CREATE TABLE AUDIO (
            ID INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            VIDEO_ID INTEGER NOT NULL,
            AUDIO_LENGTH INTEGER,
            SPEAKER_EMB BLOB,
            LANG TEXT,
            AUDIO_EMB BLOB
        );
    """)

    con.execute("""
        CREATE TABLE FACE (
            ID INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            FACE_PATH TEXT,
            VIDEO_ID INTEGER,
            FACE_NORM_PATH TEXT,
            LATENT_REP BLOB
        );
    """)

    con.execute("""
        CREATE TABLE CODE_TABLE (
            ID INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            TABLE_NAME TEXT NOT NULL ,
            COLUMN_NAME TEXT NOT NULL ,
            CODE_INTEGER INTEGER NOT NULL ,
            CODE_MEANING TEXT NOT NULL 
        );
    """)

    con.execute("""
        CREATE INDEX AUDIO_VIDEO_ID_IDX ON AUDIO (VIDEO_ID);
    """)






def insertIntoDb(file,root):
    #await asyncio.sleep(10)
    #print(file)
    sql = 'SELECT COUNT (*) FROM VIDEO WHERE VIDEO_PATH = ?'
    data = [(os.path.join(root, file))]
    result = con.execute(sql, data)

    if(result.fetchall()[0][0] == 0):
        sql = 'INSERT INTO VIDEO (VIDEO_PATH, AUDIO_PRE, FACES_PRE) VALUES (?,?,?)'
        data = [(os.path.join(root, file)),0,0]
        with con:
            con.execute(sql, data)


def insertAllIntoDb():
    for root, dirs, files in os.walk(datasetPathVideo):
        #tasks = [insertIntoDb(file,root) for file in files]
        # TODO split into chunks of say 4 files
        #await asyncio.wait(tasks)
        #await asyncio.gather(*tasks)
        for file in files:
            insertIntoDb(file,root)

#asyncio.run(insertAllIntoDb())

insertAllIntoDb()# TODO Better display of progress and make indicies

with con:
    data = con.execute("SELECT COUNT(*) FROM VIDEO")
    for row in data:
        print(str(row) + " VIDEO FILES INSERTED") 