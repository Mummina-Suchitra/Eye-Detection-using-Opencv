import cv2

#Load haar cascade classifier for eye detection
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')# all the data is stored in xml to avoid excessive memory usage

#start video capture from webcam
cap = cv2.VideoCapture(0) #0-video,1-image

while True:
    # Capture frame- by -frame
    ret, frame = cap.read()
    if not ret:
        break

    #convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect eyes in the image
    eyes = eye_cascade.detectMultiScale(gray)

    #Draw rectanges around detectd eyes
    for(ex, ey, ew, eh) in eyes:

        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    #Display the resulting frame ith rectangle around
    cv2.imshow('Eye Detection',  frame)

    #Break the loop[ if 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF ==27: #27 is the ASCII Value for Esc
        break
#Release resources 
cap.release()
cv2.destroyAllWindows()

