import cv2

cap = cv2.VideoCapture('rtsp://192.168.10.215:554/user=admin&password=&channel=1&stream=0.sdp?')

while True:
    ret, img = cap.read()
    cv2.imshow("camera", img)
    if cv2.waitKey(10) == 27: # Клавиша Esc
        break
cap.release()
cv2.destroyAllWindows()
