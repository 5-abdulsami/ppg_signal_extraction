#!/bin/bash
# extract-frames.sh - Converts video to a sequence of images using ffmpeg
# and stores them to: ../data/iphone/out<i>.png

mkdir -p ../data/phone

ffmpeg -i ../data/haseeb-recording-3.mp4 -vf fps=30 ../data/phone/out%d.png # output one image after every 33.3ms