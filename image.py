import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cnt = 0
path = "dataset/progress"
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    cnt += 1
    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',frame)
 
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite( str(cnt) + ".jpg",frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()