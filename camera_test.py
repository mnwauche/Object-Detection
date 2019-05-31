import numpy as np
import cv2
import time

cap = cv2.VideoCapture(-1)
cap.set(3,320)
cap.set(4,240)
cap.set(5,30)
cap.set(12,0)

while(True):
    cur = time.time()
    # Capture frame-by-frame
    ret, frame = cap.read()
    now = time.time()
    print(now-cur)

    # Our operations on the frame come here
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
