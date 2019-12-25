import cv2

from point_polygon import Point, is_inside
from yolo3_detection_wrapper import YoloDetection

model = YoloDetection()
input_video = 'videos/people_in_lane.mp4'
cap = cv2.VideoCapture(input_video)
desired_size = (858, 480)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 12.0, desired_size)
points = [(381, 129), (415, 125), (688, 465), (357, 464)]
polygon = [Point(x) for x in points]

while cap.isOpened():
    # try:
    ret, frame = cap.read()
    if ret:
        detections = model.predict(frame)
        for i in range(len(detections)):
            box = detections[i]
            (startX, startY, endX, endY) = box.astype("int")
            centroid = Point(((startX + endX) / 2, (startY + endY) / 2))
            if is_inside(polygon, centroid):
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

        cv2.line(frame, points[0], points[-1], (0, 255, 0), 2)
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i+1], (0, 255, 0), 2)

        cv2.imshow('frame', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    # except:
    #     continue

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
