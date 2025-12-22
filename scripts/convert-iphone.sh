#!/bin/bash
# convert-iphone.sh - Converts video `recording.MOV` to a sequence of images using ffmpeg
# and stores them to: ../data/iphone/out<i>.png

ffmpeg -i ../data/iphone-recording.MOV -vf fps=30 ../data/iphone/out%d.png # output one image every 30th of a second
