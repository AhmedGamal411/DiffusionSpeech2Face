import sqlite3 as sl
class codes:

    class audio_types:
        length_3s =3
        length_6s =6
        length_12s=12
        length_24s=24

    con = sl.connect('dataset.db')
    sql = ''' INSERT INTO CODE (TABLE, COLUMN, CODE_INTEGER, CODE_MEANING) VALUES(?,?)'''
    table = "AUDIO"
    column= "AUDIO_TYPE"
    with con:
        cur = con.cursor()
        data = [
            (table,column,audio_types.length_3s,"length_3s"),
            (table,column,audio_types.length_6s,"length_6s"),
            (table,column,audio_types.length_12s,"length_12s"),
            (table,column,audio_types.length_24s,"length_24s")]
        cur.executemany(sql, data)
        con.commit()




