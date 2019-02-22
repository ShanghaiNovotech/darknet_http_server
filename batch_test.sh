#!/bin/sh


for fn in sample_images/*; do
	echo "\n$fn"
	curl -F "file=@$fn" http://localhost:5000/upload.json
done

#echo "1.jpg "
#curl -F "file=@sample_images/1.jpg" http://localhost:5000/upload.json
#echo "\n2.jpg "
#curl -F "file=@sample_images/2.jpg" http://localhost:5000/upload.json
