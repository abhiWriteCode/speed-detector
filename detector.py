import cv2
import os
import numpy as np


config = 'weights/yolov3.cfg'
weights = 'weights/yolov3.weights'

net = cv2.dnn.readNet(weights, config)

net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


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

def detect(frame):
    frame = np.copy(frame)
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
    # print(indices)
    # return boxes, confidences, class_ids
    boxes_return, class_ids_return, confidences_return = [], [], []

    for i in indices:
        i = i[0]
        box = boxes[i]
        x, y, w, h = list(map(round, box))
        boxes_return.append([x, y, w, h])
        class_ids_return.append(class_ids[i])
        confidences_return.append(confidences[i])
        # draw_prediction(frame, class_ids[i], confidences[i], x, y, x+w, y+h)
        # pass

    return boxes_return, class_ids_return, confidences_return
