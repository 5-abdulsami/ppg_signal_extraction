#!/bin/bash
# convert-iphone.sh - Converts video `recording.MOV` to a sequence of images using ffmpeg
# and stores them to: ../data/iphone/out<i>.png

ffmpeg -i ../data/haseeb-recording-3.mp4 -vf fps=30 ../data/phone/out%d.png # output one image after every 33.3ms