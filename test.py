import cv2
import numpy as np
import time
import copy
import os
import glob
import multiprocessing as mpr
from datetime import datetime

from kalman_filter import KalmanFilter
from tracker import Tracker

config = 'weights/yolov3.cfg'
weights = 'weights/yolov3.weights'
classes_txt = 'weights/yolov3.txt'
classes = None

with open(classes_txt, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

net = cv2.dnn.readNet(weights, config)

net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

totalFrames = -1
skip_frames = 100

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    if class_id > 7: # return if object is not vehicle or pedestrian
        return
    if class_id == 0:
        color = (255,255,0) # color for pedestrian
    else:
        color = (0,255,255) # color for vehicle

    label = str(classes[class_id])
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



if __name__ == '__main__':
	FPS = 30
	'''
		Distance to line in road: ~0.025 miles
	'''
	ROAD_DIST_MILES = 0.025

	'''
		Speed limit of urban freeways in California (50-65 MPH)
	'''
	HIGHWAY_SPEED_LIMIT = 65

	# Initial background subtractor and text font
	font = cv2.FONT_HERSHEY_PLAIN

	centers = [] 

	# y-cooridinate for speed detection line
	Y_THRESH = 240

	blob_min_width_far = 6
	blob_min_height_far = 6

	blob_min_width_near = 18
	blob_min_height_near = 18

	frame_start_time = None

	# Create object tracker
	tracker = Tracker(80, 3, 2, 1)

	# Capture livestream
	# cap = cv2.VideoCapture('http://wzmedia.dot.ca.gov:1935/D3/80_whitmore_grade.stream/'+'playlist.m3u8')
	cap = cv2.VideoCapture('videos/highway.avi')

	while True:

		totalFrames += 1
		if totalFrames % skip_frames != 0:
			continue

		centers = []
		frame_start_time = datetime.utcnow()
		_, frame = cap.read()

		if frame is None:
			break

		orig_frame = copy.copy(frame)

		Width = frame.shape[1]
		Height = frame.shape[0]
	    
	    
		scale = 0.00392
		blob = cv2.dnn.blobFromImage(frame, scale, (416,416), (0,0,0), True, crop=False)
		net.setInput(blob)

		outs = net.forward(get_output_layers(net))

		class_ids = []
		confidences = []
		boxes = []
		conf_threshold = 0.5
		nms_threshold = 0.4

		for out in outs:
			for detection in out:
				scores = detection[5:]
				class_id = np.argmax(scores)
				confidence = scores[class_id]
				if confidence > 0.5:
					center_x = int(detection[0] * Width)
					center_y = int(detection[1] * Height)
					w = int(detection[2] * Width)
					h = int(detection[3] * Height)
					x = center_x - w / 2
					y = center_y - h / 2
					class_ids.append(class_id)
					confidences.append(float(confidence))
					boxes.append([x, y, w, h])


		indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)


		#  Draw line used for speed detection
		cv2.line(frame,(0, Y_THRESH),(640, Y_THRESH),(255,0,0),2)

		for i in indices:
			i = i[0]
			box = boxes[i]
			x,y,w,h = list(map(int, box))
			# draw_prediction(frame, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

			if y > Y_THRESH:
				if w >= blob_min_width_near and h >= blob_min_height_near:
					center = np.array ([[x+w/2], [y+h/2]])
					centers.append(np.round(center))

					cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
			else:
				if w >= blob_min_width_far and h >= blob_min_height_far:
					center = np.array ([[x+w/2], [y+h/2]])
					centers.append(np.round(center))

					cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

		if centers:
			tracker.update(centers)

			for vehicle in tracker.tracks:
				if len(vehicle.trace) > 1:
					for j in range(len(vehicle.trace)-1):
                        # Draw trace line
						x1 = vehicle.trace[j][0][0]
						y1 = vehicle.trace[j][1][0]
						x2 = vehicle.trace[j+1][0][0]
						y2 = vehicle.trace[j+1][1][0]

						cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

					try:
						'''
							TODO: account for load lag
						'''

						trace_i = len(vehicle.trace) - 1

						trace_x = vehicle.trace[trace_i][0][0]
						trace_y = vehicle.trace[trace_i][1][0]

						# Check if tracked object has reached the speed detection line
						if trace_y <= Y_THRESH + 5 and trace_y >= Y_THRESH - 5 and not vehicle.passed:
							cv2.putText(frame, 'I PASSED!', (int(trace_x), int(trace_y)), font, 1, (0, 255, 255), 1, cv2.LINE_AA)
							vehicle.passed = True

							load_lag = (datetime.utcnow() - frame_start_time).total_seconds()

							time_dur = (datetime.utcnow() - vehicle.start_time).total_seconds() - load_lag
							time_dur /= 60
							time_dur /= 60

							
							vehicle.mph = ROAD_DIST_MILES / time_dur

							# If calculated speed exceeds speed limit, save an image of speeding car
							if vehicle.mph > HIGHWAY_SPEED_LIMIT:
								print ('UH OH, SPEEDING!')
								cv2.circle(orig_frame, (int(trace_x), int(trace_y)), 20, (0, 0, 255), 2)
								cv2.putText(orig_frame, 'MPH: %s' % int(vehicle.mph), (int(trace_x), int(trace_y)), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
								cv2.imwrite('speeding_%s.png' % vehicle.track_id, orig_frame)
								print ('FILE SAVED!')

					
						if vehicle.passed:
							# Display speed if available
							cv2.putText(frame, 'MPH: %s' % int(vehicle.mph), (int(trace_x), int(trace_y)), font, 1, (0, 255, 255), 1, cv2.LINE_AA)
						else:
							# Otherwise, just show tracking id
							cv2.putText(frame, 'ID: '+ str(vehicle.track_id), (int(trace_x), int(trace_y)), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
					except:
						pass


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

	# remove all speeding_*.png images created in runtime
	for file in glob.glob('speeding_*.png'):
		os.remove(file)