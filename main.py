import cv2
import numpy as np
import time
# import copy
import os
import glob
import multiprocessing as mpr
from datetime import datetime

from kalman_filter import KalmanFilter
from tracker import Tracker
import detector


classes_txt = 'weights/yolov3.txt'
classes = None

with open(classes_txt, 'r') as f:
    classes = [line.strip() for line in f.readlines()]


def draw_bbox(img, class_id, x, y, w, h, Y_THRESH):
	blob_min_width_far = 6
	blob_min_height_far = 6

	blob_min_width_near = 18
	blob_min_height_near = 18

	color = None

	if y > Y_THRESH:
		if w >= blob_min_width_near and h >= blob_min_height_near:
			# center = np.array ([[x+w/2], [y+h/2]])
			# centers.append(np.round(center))
			color = (0, 0, 255)

			cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
	else:
		if w >= blob_min_width_far and h >= blob_min_height_far:
			# center = np.array ([[x+w/2], [y+h/2]])
			# centers.append(np.round(center))
			color = (0, 255, 0)

			cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

	label = str(classes[class_id])
	color = (255, 0, 0)
	cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


if __name__ == '__main__':
	FPS = 24
	'''
		Distance to line in road:
	'''
	ROAD_DIST_KMs = 0.05

	# Initial background subtractor and text font
	font = cv2.FONT_HERSHEY_PLAIN

	centers = [] 

	# y-cooridinate for speed detection line
	Y_THRESH = 240

	frame_start_time = None

	# Create object tracker
	tracker = Tracker(dist_thresh=80, max_frames_to_skip=10, 
	                  max_trace_length=2, trackIdCount=1)

	cap, first_time = cv2.VideoCapture('videos/Traffic Monitoring.mp4'), True
	# cap, first_time = cv2.VideoCapture('videos/video.mp4'), False

	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	Y_THRESH = frame_height // 2
	Y_THRESH_UP = Y_THRESH - 100
	Y_THRESH_DOWN = Y_THRESH + 100

	while True:

		if first_time:
			_x = 0
			while _x <= 300:
				_, frame = cap.read()
				_x += 1
			first_time = False

		centers = []
		frame_start_time = datetime.utcnow()
		_, frame = cap.read()

		if frame is None:
			break

		orig_frame = np.copy(frame)

		#  Draw line used for speed detection
		cv2.line(frame,(0, Y_THRESH),(frame_width, Y_THRESH),(255,0,0),2)
		boxes, class_ids, _ = detector.detect(frame)

		for box, class_id in zip(boxes, class_ids):
			x,y,w,h = box
			if class_id > 7: # those objects are irrelavant
				continue

			center = np.array([[x + w/2], [y + h/2]])
			centers.append(np.round(center))

			draw_bbox(frame, class_id, x, y, w, h, Y_THRESH)

		if centers:
			tracker.update(centers)

			for vehicle in tracker.tracks:
				if len(vehicle.trace) > 1:

					trace_i = len(vehicle.trace) - 1

					trace_x = vehicle.trace[trace_i][0][0]
					trace_y = vehicle.trace[trace_i][1][0]

					# Check if tracked object has reached the speed detection line
					if trace_y <= Y_THRESH + 5 and trace_y >= Y_THRESH - 5 and not vehicle.passed:
						cv2.putText(frame, 'I PASSED!', (int(trace_x), int(trace_y)), font, 1, (0, 255, 255), 1, cv2.LINE_AA)
						vehicle.passed = True

						current_frame_time = datetime.utcnow()
						load_lag = (current_frame_time - frame_start_time).total_seconds()
						time_dur = (current_frame_time - vehicle.start_time).total_seconds() - load_lag
						time_dur = time_dur / (60 * 60)
						
						vehicle.kph = ROAD_DIST_KMs / time_dur
						# print(vehicle.kph)
						color = (255, 0, 0)
						cv2.putText(frame, str(vehicle.kph)[:-10], (x+10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
				
					if vehicle.passed:
						# Display speed if available
						cv2.putText(frame, 'KPH: %s' % int(vehicle.kph), (int(trace_x), int(trace_y)), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
					else:
						# Otherwise, just show tracking id
						cv2.putText(frame, 'ID: '+ str(vehicle.track_id), (int(trace_x), int(trace_y)), font, 1, (0, 255, 0), 2, cv2.LINE_AA)


		# Display all images
		cv2.imshow ('original', frame)

		# Quit when escape key pressed
		key = cv2.waitKey(5)
		if key in (27, ord(' '), ord('q')):
			break

		# Sleep to keep video speed consistent
		time.sleep(1.0 / FPS)

	# Clean up
	cap.release()
	cv2.destroyAllWindows()