import cv2 as cv


cap = cv.VideoCapture(0)

#face classifier
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True: 
    ret, frame = cap.read()

    #grayscale 
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y,w, h) in faces: 
        cv.rectangle(frame, (x, y), (x +w, y +h),(0, 255, 0), 10)
        cv.putText(frame, 'Face Detect', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv.imshow("face Detect", frame)

    if cv.waitKey(1) == ord('q'):
        break


cap.release()
cv.destroyAllWindows()
