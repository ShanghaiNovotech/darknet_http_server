import sqlite3
import json
import pandas as pd
import numpy as np
import collections

REMAKE=False

con=sqlite3.connect("darknet_server.db")

minute_data = pd.read_sql_query("SELECT * FROM cam_detection_min",con)

#fix numpy sends back NaT.
def strr(obj):
    x = str(obj)
    if x=="NaT":
        return "-1.0"
    else:
        return x

if minute_data.empty:
    start_date = "2019-01-01 00:00:00"
else:
    last_row = minute_data['sampled_at'].iloc[-1]
    start_date = last_row

if REMAKE:
    start_date = "2019-01-01 00:00:00"
    sql = "DELETE FROM cam_detection_min"
    print(sql)
    rs = con.execute(sql)
    con.commit()

print(minute_data.tail(50))
print(start_date)

data=pd.read_sql_query("SELECT * FROM cam_detection WHERE created_at > '"+str(start_date)+"'" ,con)

#split into multi dataframe, by "cam_id"
for x, new_df in data.groupby("cam_id"):
    cam_id = x
    cam_name = new_df['cam_name'].iloc[-1]
    num_persons=[]
    num_luggages=[]
    sampled_ats=[]
    #iterate all rows belongs to this "cam_id"
    for index, row in new_df.iterrows():
        det=json.loads((row['det']))
        num_person=0
        num_luggage=0
        sampled_at=row['created_at']
        for elem in det:
            if elem[0] == 'person':
                num_person+=1

            if elem[0] in ['backpack','handbag']:
                num_luggage+=1

        num_persons.append(num_person)            
        num_luggages.append(num_luggage)
        sampled_ats.append(sampled_at)

    new_df.insert(1, "num_persons", num_persons, True)
    new_df.insert(1, "num_luggages", num_luggages, True)
    new_df.insert(1, "sampled_at", sampled_ats, True)

    new_df=new_df.set_index(pd.DatetimeIndex(new_df['sampled_at']))
    new_df=new_df[['num_persons', 'num_luggages']]
    down_sampled=new_df.resample('1min').mean() #resample removes 
    down_sampled=down_sampled.reset_index()

    #ditch earlier due to partial calcualtion.
    sql = "SELECT * FROM cam_detection_min WHERE cam_id="+str(cam_id)+" AND sampled_at>='"+str(down_sampled['sampled_at'].min())+"';"
    sql = "DELETE FROM cam_detection_min WHERE cam_id="+str(cam_id)+" AND sampled_at>='"+str(down_sampled['sampled_at'].min())+"';"
   
    print(sql)
    rs = con.execute(sql)
    con.commit()
   

    #pass for now
    for index, row in down_sampled.iterrows():
        sql = "INSERT INTO cam_detection_min (cam_id,cam_name,num_persons,num_luggages,sampled_at) VALUES( "+str(cam_id)+", '"+ cam_name +"', " + strr(row['num_persons'])+ ", "+ strr(row['num_luggages'])+ ",'" +str(row['sampled_at']) +"');"
        print(sql)
        rs = con.execute(sql)
        con.commit()

exit(0)
