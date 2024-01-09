import cv2


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error:camera not found")

while cap.isOpened():
    returned , frame = cap.read()

    if not returned:
        print("Error: video frame read error")
        break
    cv2.putText(frame,str("Hello, future CV engineer"),(50,50),cv2.FONT_ITALIC,1,(100,100,100),2,cv2.LINE_AA)

    cv2.imshow("camera",frame)
    if(cv2.waitKey(1) & 0xFF == ord("q")):
        break
cap.release()
cv2.destroyAllWindows()
