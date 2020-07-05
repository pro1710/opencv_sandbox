# chroma_keying

import cv2
import numpy as np

MAIN_WINDOW_NAME = 'Ball Tracker'

PATH_TO_DATA = './'
PATH_TO_MODEL_DATA = './model/'

CONFIDENCE_THRESHOLD = 0.5  
NMS_THRESHOLD = 0.4   

classes_file = PATH_TO_MODEL_DATA + "coco.names"
CLASSES = None

with open(classes_file, 'rt') as f:
    CLASSES = f.read().rstrip('\n').split('\n')

def getYoloV3():
    conf = PATH_TO_MODEL_DATA + "yolov2-tiny.cfg"
    weights = PATH_TO_MODEL_DATA + "yolov2-tiny.weights"

    return cv2.dnn.readNetFromDarknet(conf, weights)   

def getYoloV3BlobConverter():
    return lambda x: cv2.dnn.blobFromImage(x, 1/255, (416, 416), [0,0,0], 1, crop=False)

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def drawBb(frame, bb, color):
    p1 = (int(bb[0]), int(bb[1]))
    p2 = (int(bb[0] + bb[2]), int(bb[1] + bb[3]))
    cv2.rectangle(frame, p1, p2, color, 3)

def drawPredictedBb(frame, classId, conf, bb):

    left, top, width, height = bb
    right = left + width
    bottom = top + height

    drawBb(frame, bb, (255, 178, 50))

    label = '%.2f' % conf

    if CLASSES:
        assert(classId < len(CLASSES))
        label = '%s:%s' % (CLASSES[classId], label)

    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

def detectBall(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []

    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            if classId == 32:
                confidence = scores[classId]
                if confidence > CONFIDENCE_THRESHOLD:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

    if not boxes:
        return None

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    
    boxes = [boxes[i[0]] for i in indices]

    id_max = boxes.index(max(boxes, key=lambda x: x[2] * x[3]))
    max_bb = tuple(boxes[id_max])

    return max_bb

def detect(net, blob_converter, frame):
    blob = blob_converter(frame)
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))

    bb = detectBall(frame, outs)
    return bb

def getTracker(frame, bb):
    tracker = cv2.TrackerKCF_create()
    tracker_status = tracker.init(frame, bb)

    return tracker_status, tracker

def trackerFailure(frame):
    cv2.putText(frame, "Tracker failure", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2, cv2.LINE_AA)

def detectorFailure(frame):
    cv2.putText(frame, "Detector failure", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2, cv2.LINE_AA)

def main():

    cap = cv2.VideoCapture(PATH_TO_DATA + 'path_to_video_with_soccer_ball.mp4')
    _, frame = cap.read()
    
    cv2.namedWindow(MAIN_WINDOW_NAME)
    cv2.moveWindow(MAIN_WINDOW_NAME, 32, 32)

    net = getYoloV3()
    blob_converter = getYoloV3BlobConverter()

    tracker_status, tracker = None, None

    TRACKER_TTL = 10
    TRACKER_FAILURE_THRESHOLD = 10

    tracker_ttl = TRACKER_TTL
    tarcker_failures = 0

    while True:

        timer = cv2.getTickCount()
        
        _, frame = cap.read()
        if frame is None:
            break

        if not tracker: # detect the object
            # print('No tracker')

            bb = detect(net, blob_converter, frame)
            if not bb:
                # print('Detector failed')
                detectorFailure(frame)

            else:
                drawBb(frame, bb, (255, 0, 0))
                tracker_status, tracker = getTracker(frame, bb)
                tracker_ttl = TRACKER_TTL
                tracker_failures = TRACKER_FAILURE_THRESHOLD

                # print('Tracker created bb = ', bb)

        else:
            tracker_status, bb = tracker.update(frame)
            if tracker_status:
                # print('Tracker ok, bb = ', bb)
                drawBb(frame, bb, (0, 255, 0))
                tracker_failures = TRACKER_FAILURE_THRESHOLD
                tracker_ttl -= 1

                if tracker_ttl == 0:
                    tracker = None
            else:
                # print('Tracker failed')
                tracker_failures -= 1
                trackerFailure(frame)

                if tracker_failures == 0:
                    tracker = None


        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(frame, "FPS : " + str(int(fps)), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

        cv2.imshow(MAIN_WINDOW_NAME, frame)

        k = cv2.waitKey(24) & 0xff
        if k == 27: 
            break

        # print('-'*80)



        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


