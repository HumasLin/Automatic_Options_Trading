import wrds
import random
import pandas as pd
import numpy as np
from datetime import date

def query(secid,this_year,previous_time,today):
    sql_query = """
    SELECT secid,
           date,
           symbol,
           exdate,
           forward_price,
           strike_price,
           best_bid,
           best_offer,
           impl_volatility,
           delta
    FROM optionm.opprcd{}
    WHERE secid = {}
    AND cp_flag = 'C'
    AND date BETWEEN '{}' AND '{}'
    """.format(this_year,secid,previous_time,today)
    
    return sql_query

def get_data(secid):

    db = wrds.Connection()

    dfs = []
    for i in range(10,0,-1):
        today = date.today()        
        this_year = today.year-i
        previous_time = str(this_year)+'-01-01'
        today = str(this_year)+'-12-31'

        sql_query = query(secid,this_year,previous_time,today)

        db_query = db.raw_sql(sql_query)
        dfs.append(db_query)

    db.close()
    return dfs