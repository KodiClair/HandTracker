import cv2
import mediapipe as mp
import time
import math

def diffFinder(fingdiff1, fingdiff2):
    return math.sqrt(math.pow(fingdiff1, 2) + math.pow(fingdiff2, 2))


class handDetection():
    def __init__(self, mode=False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5): # defines the parameters of the class
        self.mode = mode #create an object and the object will have its own variable
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands() # add params into Hands()
        self.mpDraw = mp.solutions.drawing_utils #method to draw at points

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS) # draws the points and connections

        return img
        
    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)
                # lmList.append([id,cx,cy])
                lmList.append([cx,cy])
                if draw == True and id == 8 or id == 4:
                    cv2.circle(img, (cx, cy), 10, (245, 152, 66), cv2.FILLED)
                if draw == True and id == 12 or id == 16:
                    cv2.circle(img, (cx, cy), 10, (0, 0, 0), cv2.FILLED)

        
        return lmList
    
    def checkDraw(self, lmList):
        fingdiff1 = lmList[4][0] - lmList[8][0]
        fingdiff4 = lmList[4][1] - lmList[8][1]
        diff = diffFinder(fingdiff1, fingdiff4)
        # print(lmList[4], lmList[8], lmList[12], diff)

        if diff >= 0 and diff <= 25:
            return True
        else:
            return False
        
    def checkErase(self, lmList):
        fingdiff1 = lmList[12][0] - lmList[16][0]
        fingdiff4 = lmList[12][1] - lmList[16][1]
        diff = diffFinder(fingdiff1, fingdiff4)
        print(lmList[12], lmList[16], diff)

        if diff >= 0 and diff <= 15:
            return True
        else:
            return False

            

        
        



def main():
    pTime = 0
    pTime2 = 0
    cTime = 0
    cTime2 = 0

    cap = cv2.VideoCapture(0) # image capture object

    detector = handDetection() # instance of the class handDetector

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        change = False

        if len(lmList) != 0:
            checkKey = detector.checkSymbol(lmList)
            if checkKey:
                    cv2.putText(img, "Detected", (10,150), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
                    cv2.circle(img, lmList[4], 1, (255, 0, 0), 3, 1)
        
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3) # displays fps

        cv2.imshow("Image", img) # displays original image
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    

if __name__ == "__main__":
    main()