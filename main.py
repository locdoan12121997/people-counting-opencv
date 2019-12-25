# USAGE
# To read and write back out to video:
# python main.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4 \
#	--output output/output_01.avi
#
# To read from webcam and write back out to disk:
# python main.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
#	--output output/webcam_output.avi

import argparse
import time

import cv2
import dlib
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream

from box_drawer import BoxDrawController
from duration_estimator import StayDurationEstimator
from line_function import LineFunction
from memory_controller import MemoryController
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from yolo3_detection_wrapper import YoloDetection

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default='mobilenet_ssd/MobileNetSSD_deploy.prototxt',
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default='mobilenet_ssd/MobileNetSSD_deploy.caffemodel',
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
                help="# of skip frames between detections")
ap.add_argument("-o", "--output", default='output/output.mp4', type=str,
                help="path to optional output video file")
ap.add_argument("-t", "--trending_path", default='output/trending_output.mp4', type=str,
                help="path to optional treding output video file")
ap.add_argument("-i", "--input", default='videos/hiv00229_original.mp4', type=str,
                help="path to optional input video file")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
CENTROIDPATH = 'centroid.json'
memory_controller = MemoryController(CENTROIDPATH)
box_draw_controller = BoxDrawController()

# which direction is entrance
lines = [((0, 350), (857, 350), 'up'), ((0, 146), (857, 146), 'down')]
FRAMEPERSEC = 30.0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_size = (480, 858)
stay_estimator = StayDurationEstimator(FRAMEPERSEC)

# load our serialized model from disk
print("[INFO] loading model...")
# net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(args["input"])

# initialize the video writer (we'll instantiate later if need be)
writer = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalEnter = 0
totalExit = 0

# start the frames per second throughput estimator
fps = FPS().start()
print(FRAMEPERSEC, fourcc)
trending_video = cv2.VideoWriter(args["trending_path"], fourcc, FRAMEPERSEC,
                                 (858, 480), False)
model = YoloDetection()


# loop over frames from the video stream


frame_number = 0
trending_frame = np.zeros(frame_size)

while True:
    # grab the next frame and handle if we are reading from either
    # VideoCapture or VideoStream
    frame = vs.read()
    frame = frame[1] if args.get("input", False) else frame

    # if we are viewing a video and we did not grab a frame then we
    # have reached the end of the video
    if args["input"] is not None and frame is None:
        break

    # resize the frame to have a maximum width of 500 pixels (the
    # less data we have, the faster we can process it), then convert
    # the frame from BGR to RGB for dlib
    # frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # if we are supposed to be writing a video to disk, initialize
    # the writer
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args["output"], fourcc, FRAMEPERSEC,
                                 (W, H), True)

    # initialize the current status along with our list of bounding
    # box rectangles returned by either (1) our object detector or
    # (2) the correlation trackers
    status = "Waiting"
    rects = []

    # check to see if we should run a more computationally expensive
    # object detection method to aid our tracker
    if totalFrames % args["skip_frames"] == 0:
        # set the status and initialize our new set of object trackers
        status = "Detecting"
        trackers = []

        # convert the frame to a blob and pass the blob through the
        # network and obtain the detections
        # blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        # net.setInput(blob)
        # detections = net.forward()
        detections = model.predict(frame)

        # loop over the detections
        # for i in np.arange(0, detections.shape[2]):
        for i in range(len(detections)):
            # extract the confidence (i.e., probability) associated
            # with the prediction
            # confidence = detections[0, 0, i, 2]

            # filter out weak detections by requiring a minimum
            # confidence
            # if confidence > args["confidence"]:
            if True:
                # extract the index of the class label from the
                # detections list
                # idx = int(detections[0, 0, i, 1])

                # if the class label is not a person, ignore it
                # if CLASSES[idx] != "person":
                # 	continue

                # compute the (x, y)-coordinates of the bounding box
                # for the object
                # box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                box = detections[i]
                (startX, startY, endX, endY) = box.astype("int")

                # construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation
                # tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)

                # add the tracker to our list of trackers so we can
                # utilize it during skip frames
                trackers.append(tracker)

    # otherwise, we should utilize our object *trackers* rather than
    # object *detectors* to obtain a higher frame processing throughput
    else:
        # loop over the trackers
        for tracker in trackers:
            # set the status of our system to be 'tracking' rather
            # than 'waiting' or 'detecting'
            status = "Tracking"

            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))

    # draw a horizontal line in the center of the frame -- once an
    # object crosses this line we will determine whether they were
    # moving 'up' or 'down'
    for line in lines:
        cv2.line(frame, line[0], line[1], (0, 255, 255), 2)

    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

        if centroid[1] < trending_frame.shape[0] and centroid[0] < trending_frame.shape[1]:
            trending_frame[centroid[1], centroid[0]] = 1

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            for line in lines:
                y = [c[1] for c in to.centroids]
                x = [c[0] for c in to.centroids]
                mean_point = (np.mean(x), np.mean(y))
                updown_direction = centroid[1] - np.mean(y)
                rightleft_direction = centroid[0] - np.mean(x)
                to.centroids.append(centroid)
                memory_controller.add_centroid(objectID, centroid)

                lineFunction = LineFunction(line[0], line[1])
                # check to see if the object has been counted or not
                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                # line, count the object

                # if updown_direction

                if line[2] == 'down':
                    if updown_direction < 0 and lineFunction.fx(centroid) * lineFunction.fx(mean_point) < 0:
                        totalExit += 1
                        box_draw_controller.exit_register(objectID)
                        stay_estimator.add_frame_out(frame_number)
                        to.centroids = list()
                    # if the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object
                    elif updown_direction > 0 and lineFunction.fx(centroid) * lineFunction.fx(mean_point) < 0:
                        totalEnter += 1
                        box_draw_controller.enter_register(objectID)
                        stay_estimator.add_frame_in(frame_number)
                        to.centroids = list()

                elif line[2] == 'up':
                    if updown_direction < 0 and lineFunction.fx(centroid) * lineFunction.fx(mean_point) < 0:
                        totalEnter += 1
                        box_draw_controller.enter_register(objectID)
                        stay_estimator.add_frame_in(frame_number)
                        to.centroids = list()
                    # if the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object
                    elif updown_direction > 0 and lineFunction.fx(centroid) * lineFunction.fx(mean_point) < 0:
                        totalExit += 1
                        box_draw_controller.exit_register(objectID)
                        stay_estimator.add_frame_out(frame_number)
                        to.centroids = list()

                elif line[2] == 'left':
                    if rightleft_direction < 0 and lineFunction.fx(centroid) * lineFunction.fx(mean_point) < 0:
                        totalEnter += 1
                        box_draw_controller.enter_register(objectID)
                        stay_estimator.add_frame_in(frame_number)
                        to.centroids = list()
                    # if the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object
                    elif updown_direction > 0 and lineFunction.fx(centroid) * lineFunction.fx(mean_point) < 0:
                        totalExit += 1
                        box_draw_controller.exit_register(objectID)
                        stay_estimator.add_frame_out(frame_number)
                        to.centroids = list()

                elif line[2] == 'right':
                    if rightleft_direction < 0 and lineFunction.fx(centroid) * lineFunction.fx(mean_point) < 0:
                        totalExit += 1
                        box_draw_controller.exit_register(objectID)
                        stay_estimator.add_frame_out(frame_number)
                        to.centroids = list()
                    # if the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object
                    elif updown_direction > 0 and lineFunction.fx(centroid) * lineFunction.fx(mean_point) < 0:
                        totalEnter += 1
                        box_draw_controller.enter_register(objectID)
                        stay_estimator.add_frame_in(frame_number)
                        to.centroids = list()

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        box_width = 10
        box_height = 30
        enter_object = box_draw_controller.get_enter_object_list()
        if enter_object and objectID in enter_object:
            text = "ID {}".format(objectID)
            # cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            # 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            startX = centroid[0] - box_width
            startY = centroid[1] - box_height
            endX = centroid[0] + box_width
            endY = centroid[1] + box_height
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        # cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        exit_object = box_draw_controller.get_exit_object_list()
        if exit_object and objectID in exit_object:
            text = "ID {}".format(objectID)
            # cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            # 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            startX = centroid[0] - box_width
            startY = centroid[1] - box_height
            endX = centroid[0] + box_width
            endY = centroid[1] + box_height
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
        # cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
    box_draw_controller.degenerate()

    # construct a tuple of information we will be displaying on the
    # frame
    info = [
        ("Exit", totalExit),
        ("Enter", totalEnter),
        # ("Status", status),
        ("Duration", stay_estimator.calculate_average_duration())
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # check to see if we should write the frame to disk
    if writer is not None:
        writer.write(frame)
    trending_video.write(np.uint8(trending_frame * 255))

    # show the output frame
    # cv2.imshow("Frame", frame)
    cv2.imshow("Frame", trending_frame * 255)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # increment the total number of frames processed thus far and
    # then update the FPS counter
    totalFrames += 1
    fps.update()
    frame_number += 1

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if writer is not None:
    writer.release()

# if we are not using a video file, stop the camera video stream
if not args.get("input", False):
    vs.stop()

# otherwise, release the video file pointer
else:
    vs.release()

cv2.imwrite("output/trending.jpg", trending_frame)
# close any open windows
trending_video.release()
cv2.destroyAllWindows()
memory_controller.close()
