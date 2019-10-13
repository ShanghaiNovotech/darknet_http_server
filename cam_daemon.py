import requests
import time
import os

url = 'http://192.168.9.78:8080/capture/webCapture.jpg?channel=1&FTPsend=0&checkinfo=0'
url = 'https://picsum.photos/536/354'
SRV_URL='http://localhost:5000/upload'
PID=os.getpid()

def fetch_n_post(url):
    date_string = time.strftime("%Y%m%d-%H%M")
    fn=date_string+".jpg"
    fn="/tmp/"+str(PID)+"cam_ftech.jpg"
    fn2=date_string+".txt"
    res1 = requests.get(url)
    if res1.status_code == 200:
        with open(fn, 'wb') as f:
            f.write(res1.content)

    #post fn to server
    files = {'file': open(fn, 'rb')}
    res2 = requests.post(SRV_URL, files=files, data={"cam_id":1, "cam_name":"random"})
    if res2.status_code == 200:
        print(res2.content)
        '''
        with open(fn2, 'wb') as f:
            f.write(res2.content)
            print(res2.content)
        '''

while True:
    try:
        fetch_n_post(url)
    except:
        print("EXCEPTION in fetch_n_post")
        pass

    time.sleep(1)
