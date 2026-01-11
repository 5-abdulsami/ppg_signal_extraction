#!/bin/bash
# extract-frames.sh - Converts video to a sequence of images using ffmpeg
# and stores them to: ../data/frames/out<i>.png

mkdir -p ../data/frames

ffmpeg -i ../data/recordings/haseeb-recording-3.mp4 -vf fps=30 ../data/frames/out%d.png # output one image after every 33.3ms