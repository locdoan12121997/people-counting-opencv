import cv2

input_video = 'videos/sml23_CAM3_70.mp4'
cap = cv2.VideoCapture(input_video)

desired_size = (1920, 1080)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output3.mp4',fourcc, 25.0, desired_size)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, desired_size)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
