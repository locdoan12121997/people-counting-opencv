import cv2

input_video = 'videos/hiv00229.mp4'
cap = cv2.VideoCapture(input_video)

desired_size = (858, 480)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output4.mp4',fourcc, 30.0, desired_size)

while(cap.isOpened()):
    try:
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
    except:
        continue

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
