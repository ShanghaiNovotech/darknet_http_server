import datetime
import sqlite3
import json
import time
import pandas as pd
import numpy as np
import collections

REMAKE = False
HARD_START_DATE = "2019-01-01 00:00:00"
con = sqlite3.connect("darknet_server.db")


# fix numpy sends back NaT.
def strr(obj):  # we can use fillna
    x = str(obj)
    if x == "NaT":
        return "-1.0"
    else:
        return x


def calc_min_data():
    minute_data = pd.read_sql_query("SELECT * FROM cam_detection_min order by sampled_at ASC", con)

    if minute_data.empty:
        start_date = HARD_START_DATE
    else:
        last_row = minute_data['sampled_at'].iloc[-1]
        start_date = last_row

    if REMAKE:
        start_date = HARD_START_DATE
        sql = "DELETE FROM cam_detection_min"
        print(sql)
        rs = con.execute(sql)
        con.commit()

    print(minute_data.tail(50))
    print(start_date)
    data = pd.read_sql_query("SELECT * FROM cam_detection WHERE created_at > '" + str(start_date) + "'", con)

    # split into multi dataframe, by "cam_id"
    for x, new_df in data.groupby("cam_id"):
        cam_id = x
        cam_name = new_df['cam_name'].iloc[-1]
        num_persons = []
        num_luggages = []
        sampled_ats = []
        # iterate all rows belongs to this "cam_id"
        for index, row in new_df.iterrows():
            det = json.loads((row['det']))
            num_person = 0
            num_luggage = 0
            sampled_at = row['created_at']
            for elem in det:
                if elem[0] == 'person':
                    num_person += 1

                if elem[0] in ['backpack', 'handbag', 'suitcase']:
                    num_luggage += 1

            num_persons.append(num_person)
            num_luggages.append(num_luggage)
            sampled_ats.append(sampled_at)

        new_df.insert(1, "num_persons", num_persons, True)
        new_df.insert(1, "num_luggages", num_luggages, True)
        new_df.insert(1, "sampled_at", sampled_ats, True)

        new_df = new_df.set_index(pd.DatetimeIndex(new_df['sampled_at']))
        new_df = new_df[['num_persons', 'num_luggages']]
        down_sampled = new_df.resample('1min').mean()  # resample removes
        down_sampled = down_sampled.reset_index()

        # ditch earlier due to partial calcualtion.
        sql = "SELECT * FROM cam_detection_min WHERE cam_id=" + str(cam_id) + " AND sampled_at>='" + str(
            down_sampled['sampled_at'].min()) + "';"
        sql = "DELETE FROM cam_detection_min WHERE cam_id=" + str(cam_id) + " AND sampled_at>='" + str(
            down_sampled['sampled_at'].min()) + "';"

        print(sql)
        rs = con.execute(sql)
        con.commit()

        # pass for now
        for index, row in down_sampled.iterrows():
            sql = "INSERT INTO cam_detection_min (cam_id,cam_name,num_persons,num_luggages,sampled_at) VALUES( " + str(
                cam_id) + ", '" + cam_name + "', " + strr(row['num_persons']) + ", " + strr(
                row['num_luggages']) + ",'" + str(row['sampled_at']) + "');"
            print(sql)
            rs = con.execute(sql)
            con.commit()


def calc_hour_data():
    hour_data = pd.read_sql_query("SELECT * FROM cam_detection_hour order by sampled_at ASC", con)
    if hour_data.empty:
        start_date = "2019-10-01 00:00:00"
    else:
        last_row = hour_data['sampled_at'].iloc[-1]
        start_date = last_row
    data = pd.read_sql_query("SELECT * FROM cam_detection WHERE created_at > '" + str(start_date) + "'", con)

    for x, new_df in data.groupby("cam_id"):
        cam_id = x
        cam_name = new_df['cam_name'].iloc[-1]
        num_persons = []
        num_luggages = []
        sampled_ats = []
        # iterate all rows belongs to this "cam_id"
        for index, row in new_df.iterrows():
            det = json.loads((row['det']))
            num_person = 0
            num_luggage = 0
            sampled_at = row['created_at']
            for elem in det:
                if elem[0] == 'person':
                    num_person += 1

                if elem[0] in ['backpack', 'handbag', 'suitcase']:
                    num_luggage += 1
            num_persons.append(num_person)
            num_luggages.append(num_luggage)
            sampled_ats.append(sampled_at)
        new_df.insert(1, "num_persons", num_persons, True)

        new_df.insert(1, "num_luggages", num_luggages, True)

        new_df.insert(1, "sampled_at", sampled_ats, True)

        new_df = new_df.set_index(pd.DatetimeIndex(new_df['sampled_at']))

        new_df = new_df[['num_persons', 'num_luggages']]

        down_sampled = new_df.resample('1h').sum()  # resample removes

        down_sampled = down_sampled.reset_index()
        sql = "DELETE FROM cam_detection_hour WHERE cam_id=" + str(cam_id) + " AND sampled_at>='" + str(
            down_sampled['sampled_at'].min()) + "';"

        rs = con.execute(sql)
        con.commit()

        for index, row in down_sampled.iterrows():
            sql = "INSERT INTO cam_detection_hour (cam_id,cam_name,num_persons,num_luggages,sampled_at) VALUES( " + str(
                cam_id) + ", '" + cam_name + "', " + strr(row['num_persons']) + ", " + strr(
                row['num_luggages']) + ",'" + str(row['sampled_at']) + "');"

            rs = con.execute(sql)
            con.commit()


if __name__ == "__main__":
    while True:
        calc_min_data()
        print("LOOP done, sleep 20 sec.")
        time.sleep(20)
        time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if time_now[14:16] == '01':
            print("update cam_detection_hour")
            calc_hour_data()
