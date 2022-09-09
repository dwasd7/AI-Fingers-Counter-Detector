
import cv2
from cvzone.PoseModule import PoseDetector

bodydetector = PoseDetector()
cap = cv2.VideoCapture(0)
while True:
    success, img =cap.read()
    img = bodydetector.findPose(img)
    dots , bbox = bodydetector.findPosition(img, bboxWithHands=True)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
