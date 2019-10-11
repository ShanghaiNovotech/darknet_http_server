import requests
import time

url = 'http://192.168.9.78:8080/capture/webCapture.jpg?channel=1&FTPsend=0&checkinfo=0'
SRV_URL='http://localhost:5000/upload'

def fetech_image_and_post(url):
    date_string = time.strftime("%Y%m%d-%H%M")
    fn=date_string+".jpg"
    fn="cam_ftech.jpg"
    fn2=date_string+".txt"
    res1 = requests.get(url)
    if res1.status_code == 200:
        with open(fn, 'wb') as f:
            f.write(res1.content)
            print("FETCH OK")

    #post fn to server
    files = {'file': open(fn, 'rb')}
    res2 = requests.post(SRV_URL, files=files)
    if res2.status_code == 200:
        with open(fn2, 'wb') as f:
            #f.write(res2.content)
            print(res2.content)

while True:
    fetech_image_and_post(url)
