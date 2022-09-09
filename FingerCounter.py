from sre_constants import SUCCESS
import cv2
import os
import mediapipe as mp

#I IMPLEMENT THIS PRECREATED CLASS INTO MY CODE
class HandTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils 

    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB) 
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,
                                            self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self,img, handNo=0, draw=True):
        lmlist = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h) 
                lmlist.append([id,cx,cy])
            if draw:
                cv2.circle(img,(cx,cy), 15 , (255,0,255), cv2.FILLED)

        return lmlist



Cam_Width = 800
Cam_Height = 600
capture = cv2.VideoCapture(0)
capture.set(4, Cam_Width)
capture.set(4, Cam_Height)


imgList = os.listdir("num_imgs")
print(imgList)
camList =[]

for eachimage in imgList:
    num_Image =cv2.imread(f'num_imgs/{eachimage}')
    camList.append(num_Image)


handdetector = HandTracker(detectionCon =0.75)
MPFingerTipIndex = [4,8,12,16,20]

while True: 
    success, img = capture.read()
    img = handdetector.findHands(img)
    FingerIndexposList =handdetector.findPosition(img, draw=False)  #get the positioning of the fingers accoring to mediapipe


    if len(FingerIndexposList) != 0:
        fingers = []
        fingers_name =[]

        if FingerIndexposList[MPFingerTipIndex[0]-1][1] < FingerIndexposList[MPFingerTipIndex[0]][1]:
            fingers.append(1)
            fingers_name.append("Thumb")
        else:
            fingers.append(0)


        for id in range(1,5):
            if FingerIndexposList[MPFingerTipIndex[id]][2] < FingerIndexposList[MPFingerTipIndex[id]-2][2]:
                fingers.append(1)
                if id == 1:
                    fingers_name.append("Index")
                elif id == 2:
                    fingers_name.append("Middle")
                elif id == 3:
                    fingers_name.append("Ring")
                elif id == 4:
                    fingers_name.append("Pinky")
            else:
                fingers.append(0)
        cv2.putText(img,str(', '.join(fingers_name)), (240,40),cv2.FONT_HERSHEY_PLAIN, 1, (15,0,0), 2)
        all_fingers = fingers.count(1)
        img[0:200,0:200] = camList[all_fingers-1]
    
    if not success: break
    cv2.imshow("Camera View", img)
    cv2.waitKey(1)


