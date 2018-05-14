#https://pythonprogramming.net/loading-video-python-opencv-tutorial/
import numpy
import cv2

# opens the webcame from computer
cap = cv2.VideoCapture(0)

while(True):
    # ret is a boolean regarding whether or not there was a return at all, at the frame is each frame that is returned
    ret, frame = cap.read()
    
    # frame converted to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)
    # if the key 'q' is pressed 
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
        
cap.release()
cv2.destroyAllWindows()

