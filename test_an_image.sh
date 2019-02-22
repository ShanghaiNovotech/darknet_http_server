#!/bin/sh
curl -F "file=@$1" http://localhost:5000/upload.json
