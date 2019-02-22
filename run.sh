#!/bin/sh
cd ..

FLASK_APP=darknet_http_server/darknet_flask.py flask run --host=0.0.0.0
