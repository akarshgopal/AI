import cv2
import sys

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
	
	video_capture = cv2.VideoCapture(0)

	# Capture frame-by-frame
	ret, frame = video_capture.read()

	# Display the resulting frame
	cv2.imshow('Video', frame)