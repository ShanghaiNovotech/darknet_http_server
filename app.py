#!python3
"""
Python flask server for identifying objects in images

darknet used = https://github.com/AlexeyAB/darknet

Requires DLL compilation

Both the GPU and no-GPU version should be compiled; the no-GPU version should be renamed "yolo_cpp_dll_nogpu.dll".

On a GPU system, you can force CPU evaluation by any of:

- Set global variable DARKNET_FORCE_CPU to True
- Set environment variable CUDA_VISIBLE_DEVICES to -1
- Set environment variable "FORCE_CPU" to "true"

To use, either run performDetect() after import, or modify the end of this file.

See the docstring of performDetect() for parameters.

Directly viewing or returning bounding-boxed images requires scikit-image to be installed (`pip install scikit-image`)

"""
#pylint: disable=R, W0401, W0614, W0703
from ctypes import *
import math
import random
import os
import json
import os
import cv2
import time
import shutil
import datetime
from flask import jsonify
from flask_bootstrap import Bootstrap
from flask import Flask, render_template, request, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy


#some global names
netMain = None
metaMain = None
altNames = None

#init app
app = Flask(__name__)

#load conf
with open('config.json') as f:
    config = json.load(f)
app.config.update(config)

Bootstrap(app)
db=SQLAlchemy(app)

#db classes
class CAMDetection(db.Model):
    #__tablename__ = 'detection'
    id = db.Column("id", db.Integer, primary_key=True)
    cam_id   = db.Column("cam_id", db.Integer)
    cam_name = db.Column("cam_name", db.String(16))
    pic_path = db.Column("pic_path", db.String(64))
    det = db.Column('det', db.Unicode)
    created_at = db.Column('created_at', db.DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return '<Detection %r>' % self.cam_name

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

db.create_all()
db.session.commit()


###functions and classes
def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

def network_width(net):
    return lib.network_width(net)

def network_height(net):
    return lib.network_height(net)

def array_to_image(arr):
    import numpy as np
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        if altNames is None:
            nameTag = meta.names[i]
        else:
            nameTag = altNames[i]
        res.append((nameTag, out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45, debug= False):
    """
    Performs the meat of the detection
    """
    #pylint: disable= C0321
    im = load_image(image, 0, 0)
    if debug: print("Loaded image")
    ret = detect_image(net, meta, im, thresh, hier_thresh, nms, debug)
    free_image(im)
    if debug: print("freed image")
    return ret

def detect_image(net, meta, im, thresh=.5, hier_thresh=.5, nms=.45, debug= False):
    #import cv2
    #custom_image_bgr = cv2.imread(image) # use: detect(,,imagePath,)
    #custom_image = cv2.cvtColor(custom_image_bgr, cv2.COLOR_BGR2RGB)
    #custom_image = cv2.resize(custom_image,(lib.network_width(net), lib.network_height(net)), interpolation = cv2.INTER_LINEAR)
    #import scipy.misc
    #custom_image = scipy.misc.imread(image)
    #im, arr = array_to_image(custom_image)     # you should comment line below: free_image(im)
    num = c_int(0)
    if debug: print("Assigned num")
    pnum = pointer(num)
    if debug: print("Assigned pnum")
    predict_image(net, im)
    #print("c2h2: image shape:", im.shape)
    if debug: print("did prediction")
    #dets = get_network_boxes(net, custom_image_bgr.shape[1], custom_image_bgr.shape[0], thresh, hier_thresh, None, 0, pnum, 0) # OpenCV
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, 0)
    if debug: print("Got dets")
    num = pnum[0]
    if debug: print("got zeroth index of pnum")
    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    if debug: print("did sort")
    res = []
    if debug: print("about to range")
    for j in range(num):
        if debug: print("Ranging on "+str(j)+" of "+str(num))
        if debug: print("Classes: "+str(meta), meta.classes, meta.names)
        for i in range(meta.classes):
            if debug: print("Class-ranging on "+str(i)+" of "+str(meta.classes)+"= "+str(dets[j].prob[i]))
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                if altNames is None:
                    nameTag = meta.names[i]
                else:
                    nameTag = altNames[i]
                if debug:
                    print("Got bbox", b)
                    print(nameTag)
                    print(dets[j].prob[i])
                    print((b.x, b.y, b.w, b.h))
                res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    if debug: print("did range")
    res = sorted(res, key=lambda x: -x[1])
    if debug: print("did sort")
    free_detections(dets, num)
    if debug: print("freed detections")
    return res

def performDetect(imagePath="./data/dog.jpg", thresh= 0.25, configPath = "../cfg/yolov3.cfg", weightPath = "../yolov3.weights", metaPath= "../cfg/coco_http.data", showImage= False, makeImageOnly = False, initOnly= False):
    """
    Convenience function to handle the detection and returns of objects.

    Displaying bounding boxes requires libraries scikit-image and numpy

    Parameters
    ----------------
    imagePath: str
        Path to the image to evaluate. Raises ValueError if not found

    thresh: float (default= 0.25)
        The detection threshold

    configPath: str
        Path to the configuration file. Raises ValueError if not found

    weightPath: str
        Path to the weights file. Raises ValueError if not found

    metaPath: str
        Path to the data file. Raises ValueError if not found

    showImage: bool (default= True)
        Compute (and show) bounding boxes. Changes return.

    makeImageOnly: bool (default= False)
        If showImage is True, this won't actually *show* the image, but will create the array and return it.

    initOnly: bool (default= False)
        Only initialize globals. Don't actually run a prediction.

    Returns
    ----------------------


    When showImage is False, list of tuples like
        ('obj_label', confidence, (bounding_box_x_px, bounding_box_y_px, bounding_box_width_px, bounding_box_height_px))
        The X and Y coordinates are from the center of the bounding box. Subtract half the width or height to get the lower corner.

    Otherwise, a dict with
        {
            "detections": as above
            "image": a numpy array representing an image, compatible with scikit-image
            "caption": an image caption
        }
    """
    # Import the global variables. This lets us instance Darknet once, then just call performDetect() again without instancing again
    global metaMain, netMain, altNames #pylint: disable=W0603
    assert 0 < thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `"+os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `"+os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `"+os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = load_meta(metaPath.encode("ascii"))
    if altNames is None:
        # In Python 3, the metafile default access craps out on Windows (but not Linux)
        # Read the names file and create a list to feed to detect
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    if initOnly:
        print("Initialized detector")
        return None

#allowd upload files
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in set(app.config['ALLOWED_EXTENSIONS'])


if app.config['LOAD_NNET']:
    lib = CDLL("../libdarknet.so", RTLD_GLOBAL)
    lib.network_width.argtypes = [c_void_p]
    lib.network_width.restype = c_int
    lib.network_height.argtypes = [c_void_p]
    lib.network_height.restype = c_int

    copy_image_from_bytes = lib.copy_image_from_bytes
    copy_image_from_bytes.argtypes = [IMAGE,c_char_p]


    predict = lib.network_predict_ptr
    predict.argtypes = [c_void_p, POINTER(c_float)]
    predict.restype = POINTER(c_float)

    if app.config['hasGPU']:
        set_gpu = lib.cuda_set_device
        set_gpu.argtypes = [c_int]

    make_image = lib.make_image
    make_image.argtypes = [c_int, c_int, c_int]
    make_image.restype = IMAGE

    get_network_boxes = lib.get_network_boxes
    get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
    get_network_boxes.restype = POINTER(DETECTION)

    make_network_boxes = lib.make_network_boxes
    make_network_boxes.argtypes = [c_void_p]
    make_network_boxes.restype = POINTER(DETECTION)

    free_detections = lib.free_detections
    free_detections.argtypes = [POINTER(DETECTION), c_int]

    free_ptrs = lib.free_ptrs
    free_ptrs.argtypes = [POINTER(c_void_p), c_int]

    network_predict = lib.network_predict_ptr
    network_predict.argtypes = [c_void_p, POINTER(c_float)]

    reset_rnn = lib.reset_rnn
    reset_rnn.argtypes = [c_void_p]

    load_net = lib.load_network
    load_net.argtypes = [c_char_p, c_char_p, c_int]
    load_net.restype = c_void_p

    load_net_custom = lib.load_network_custom
    load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
    load_net_custom.restype = c_void_p

    do_nms_obj = lib.do_nms_obj
    do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

    do_nms_sort = lib.do_nms_sort
    do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

    free_image = lib.free_image
    free_image.argtypes = [IMAGE]

    letterbox_image = lib.letterbox_image
    letterbox_image.argtypes = [IMAGE, c_int, c_int]
    letterbox_image.restype = IMAGE

    load_meta = lib.get_metadata
    lib.get_metadata.argtypes = [c_char_p]
    lib.get_metadata.restype = METADATA

    load_image = lib.load_image_color
    load_image.argtypes = [c_char_p, c_int, c_int]
    load_image.restype = IMAGE

    rgbgr_image = lib.rgbgr_image
    rgbgr_image.argtypes = [IMAGE]

    predict_image = lib.network_predict_image
    predict_image.argtypes = [c_void_p, IMAGE]
    predict_image.restype = POINTER(c_float)

    performDetect()

#testing if works!
#imagePath="./data/dog.jpg"
#detections = detect(netMain, metaMain, imagePath.encode("ascii"), app.config['default_thresh'])
#print("print " + json.dumps(detections))

#root
@app.route('/')
def hello_world():
    return render_template('index.html', value=0)

@app.route('/api/camera_detection')
def camera_detections():
    results=db.session.query(CAMDetection).order_by(CAMDetection.id.desc()).limit(100).all()
    ret = []
    for cam in results:
        ret.append(cam.as_dict())

    return jsonify(ret)

@app.route('/api/health')
def health():
    return ""

#return urls
@app.route('/api/latest_images')
def latest_images():
    return ""

#serve upload files
@app.route('/upload/<path:path>')
def send_js(path):
    return send_from_directory('upload', path)


#upload
@app.route('/upload', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        data = json.load(request.files['data'])
        _cam_id = data["cam_id"]
        _cam_name = data["cam_name"]
        print(_cam_id, _cam_name)
        
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            date_string = time.strftime("%Y%m%d-%H%M%S")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], "cam0_"+date_string+".jpg" )
            latest_file_path = os.path.join(app.config['UPLOAD_FOLDER'], "cam0_latest.jpg")
            file.save(file_path)
            detections = detect(netMain, metaMain, file_path.encode("ascii"), app.config['default_thresh'])
            json_detections=json.dumps(detections)

            #db save
            obj=CAMDetection(cam_id=_cam_id, cam_name=_cam_name, det=json_detections, pic_path=file_path)
            db.session.add(obj)   
            db.session.commit()

            #html processing
            res = '<pre>' + json_detections +'</pre>'
            image_html='<img src="'+file_path+'" alt="detection">'

            #annotate
            img = cv2.imread(file_path)
            img_height, img_width, img_ch = img.shape
            if(img_height > img_width):
                img_length = img_height
            else:
                img_length = img_width

            idx=0
            crop_html =""
            img2 = img.copy()
            for det in detections:
                idx += 1 
                print(json.dumps(det))
                x0 = int(det[2][0] - det[2][2] / 2)
                x1 = int(det[2][0] + det[2][2] / 2)
                y0 = int(det[2][1] - det[2][3] / 2)
                y1 = int(det[2][1] + det[2][3] / 2)
                cv2.rectangle(img,(x0, y0), (x1,y1),(0,255,0),1)
                annotation = det[0] +" "+ str(int(det[1]*100))+"%" 
                if app.config['approx_dist'] and det[0]=="person":
                    annotation = annotation+", "+str(int(2.0/(det[2][2]/float(img_length)))) + "m"

                if app.config['crop_person'] and det[0]=="person":
                    crop_img = img2[y0:y1, x0:x1].copy()
                    crop_fp= file_path+str(idx)+".png"
                    cv2.imwrite(crop_fp, crop_img)
                    crop_html += '<img src="'+crop_fp+'" alt="crop">'


                cv2.putText(img, annotation, (x0,y0+12),cv2.FONT_HERSHEY_PLAIN,1,(230,230,230),1)
            
            cv2.imwrite(file_path,img)
            shutil.copyfile(file_path, latest_file_path)

            if app.config['crop_person']:
                image_html += crop_html

            return res + image_html

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

@app.route('/api/cameras')
def api_cameras():
    return jsonify(({"cameras":app.config["cameras"]}))

#upload
@app.route('/upload.json', methods=['POST'])
def upload_json():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return "INVALID FILE"
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(file_path)
            file.save(file_path)
            detections = detect(netMain, metaMain, file_path.encode("ascii"), app.config['default_thresh'])
            return json.dumps(detections) 

    return "NO FILES UPLOADED."


#upload and split into 9 sub images (expirmental for supersized image)
@app.route('/upload9', methods=['GET','POST'])
def upload9():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(file_path)
            file.save(file_path)
            
            #split into 9 images
            img = cv2.imread(file_path)

            height, width, channels = img.shape

            x_crops=3
            y_crops=3

            print(height, width, channels)

            sub_image_paths=[]
            sub_image_dets=[]
            h = int(height / y_crops)
            w = int(width / x_crops)
            for j in range(y_crops):
                for i in range(x_crops):
                    x = int(width / x_crops * i)
                    y = int(height / y_crops * j)
                    print(x,y,h,w)
                    sub_img = img[y:y+h, x:x+w]
                    sub_img_path = file_path+"_" + str(j) + "_" + str(i) +".jpg"
                    sub_image_paths.append(sub_img_path)
                    cv2.imwrite(sub_img_path, sub_img)
                    sub_image_dets.append( detect(netMain, metaMain, sub_img_path.encode("ascii"), app.config['default_thresh']) )

            total_det = detect(netMain, metaMain, file_path.encode("ascii"), app.config['default_thresh']) 
            html = json.dumps(total_det) 
            html = html +'\n<br>'+ str(json_stats(json.dumps(total_det)) )
            html= html + '\n<br><img src="'+file_path+'" width="750" alt="detection">'
            for i in range(x_crops*y_crops):
                html=html+ '\n<br><img src="'+sub_image_paths[i]+'" width="250" alt="detection"><br><p>'+json.dumps(sub_image_dets[i])
            return html

    return '''
    <!doctype html>
    <title>Upload new File9</title>
    <h1>Upload new File9</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

def json_stats(j):
    objs=json.loads(j)
    count = len(objs)
    dets = {}
    for obj in objs:
       if dets.get(obj[0]):
           dets[obj[0]]=dets[obj[0]]+1
       else:
           dets[obj[0]]=1
    return dets


#server static image
@app.route('/darknet_http_server/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


#server upload image
@app.route('/darknet_http_server/upload/<path:path>')
def send_upload(path):
    return send_from_directory('upload', path)

