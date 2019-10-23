import requests
import time
import os
import json


with open('config.json') as f:
        config = json.load(f)

CAMS=config['cameras']

SRV_URL=config['UPLOAD_URL']
print(SRV_URL)
PID=os.getpid()
cnt=0

def fetch_n_post(_id, name, url):
    global cnt
    date_string = time.strftime("%Y%m%d-%H%M")
    fn="/tmp/"+str(PID)+"_"+ str(name) + "_cam_ftech.jpg"
    #print(fn)
    fn2=date_string+".txt"
    res1 = requests.get(url)
    if res1.status_code == 200:
        print(str(cnt) + " 200 " + url)
        cnt =  cnt + 1
        with open(fn, 'wb') as f:
            f.write(res1.content)

    #post fn to server
    data = {"cam_id":_id, "cam_name":name}
    files = {'file': open(fn, 'rb')}
    files = [
                ('file', (fn, open(fn, 'rb'), 'application/octet')),
                ('data', ('data', json.dumps(data), 'application/json')),]

    #res2 = requests.post(SRV_URL, files=files, data=json.dumps({"cam_id":_id, "cam_name":name}))
    res2 = requests.post(SRV_URL, files=files)
    if res2.status_code == 200:
        print(res2.content)
        '''
        with open(fn2, 'wb') as f:
            f.write(res2.content)
            print(res2.content)
        '''


while True:
    for cam in CAMS:
        if cam['method']=="HTTP":
            try:
                fetch_n_post(cam['id'], cam['name'], cam['uri'])

            except Exception as e: 
                print(e)

    time.sleep(1)
