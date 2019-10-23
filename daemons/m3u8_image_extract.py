import cv2
import threading
import queue
import m3u8
import requests
import shutil
import tempfile
import os
import posixpath
import urllib.parse
import re
import time
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

def is_url(uri):
    return re.match(r'https?://', uri) is not None

class SegmentDownloadThread(threading.Thread):
    def __init__(self, downloadqueue, location, total):
        threading.Thread.__init__(self)
        self.downloadQueue = downloadqueue
        self.location = location
        self.total = total
    
    def run(self):
        while True:
            item = self.downloadQueue.get()
            if item is None:
                return
                
            self.execute(item)
            self.downloadQueue.task_done()
    
    def execute(self, item):
        if item[1]:
            url = item[1] + "/" + item[2]
        else:
            url = item[2]
            item[2] = os.path.basename(urllib.parse.urlparse(url).path)
            
        if item[3]:
            backend = default_backend()
            r = requests.get(item[3].uri)
            key = r.content
            cipher = Cipher(algorithms.AES(key), modes.CBC(bytes.fromhex(item[3].iv[2:])), backend=backend)
            decryptor = cipher.decryptor()
            
        r = requests.get(url, stream=True)
        with open(os.path.join(self.location, item[2]), 'wb') as f:
            if r.status_code == 200:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        if item[3]:
                            f.write(decryptor.update(chunk))
                        else:
                            f.write(chunk)

def concatenate_files(filelist, source, destination, name):
    with open(os.path.join(destination, name), 'ab') as outfile:
        for file in filelist:
            with open(os.path.join(source, file), 'rb') as readfile:
                shutil.copyfileobj(readfile, outfile)

    outfile.close()

def m3u8_load(uri):
    r = requests.get(uri)
    m3u8_obj = m3u8.M3U8(r.text)
    return m3u8_obj

def hls_fetch(playlist_location, storage_location, download_location, localFile, loop_download):
    oldSegmentList = list()
    
    while True:
        download_queue = queue.Queue()
        tsSliceList = list()
        playlist = m3u8_load(playlist_location)
        parsed_url = urllib.parse.urlparse(playlist_location)
        prefix = parsed_url.scheme + '://' + parsed_url.netloc
        base_path = posixpath.normpath(parsed_url.path + '/..')
        base_uri = urllib.parse.urljoin(prefix, base_path)
        pool = list()
        total = 0
        
        for number, file in enumerate(playlist.segments):
            if not is_url(file.uri):
                playlist.base_uri = base_uri
            
            if file.uri not in oldSegmentList:
                total += 1
                print('Multithreading Downloading: ',file.uri)
                oldSegmentList.append(file.uri)
                tsSliceList.append(file.uri)
                download_queue.put([number, playlist.base_uri, file.uri, file.key])

        if total == 0:
            time.sleep(3)
            continue

        for i in range(total):
            thread = SegmentDownloadThread(download_queue, download_location, total)
            thread.daemon = True
            thread.start()
            pool.append(thread)
        download_queue.join()
        
        for i in range(total):
            download_queue.put(None)
        
        for thread in pool:
            thread.join()

        if not loop_download:
            os.remove(localFile)
        
        print('Concat files to: ' + localFile)
        concatenate_files(tsSliceList, download_location, storage_location, localFile)
        tsSliceList.clear()

        if not loop_download:
            return

    
def get_frame(vid_file, out_file, nframe):
    vidcap = cv2.VideoCapture(vid_file)
    success = True
    count = 0
    while success:
        success, image = vidcap.read()
        if nframe == count:
            cv2.imwrite(out_file, image)
            print("Frame  %d written as %s" % (count, out_file))
            break

        count += 1




def get_m3u8_frame(m3u8_uri, local_file, nframe):
    cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as download_location:
        print("TEMP DIR = "+download_location)
        hls_fetch(m3u8_uri, cwd, download_location, local_file, False)
        get_frame(local_file, "test.jpg", nframe)


if __name__ == "__main__":
    get_m3u8_frame("http://ivi.bupt.edu.cn/hls/cctv1hd.m3u8", "video.ts", 50)
    