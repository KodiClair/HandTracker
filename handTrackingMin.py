import cv2
import mediapipe as mp
import time

########################################Basic Code to run a webcam
cap = cv2.VideoCapture(0) # image capture object

mpHands = mp.solutions.hands
hands = mpHands.Hands()

pTime = 0
cTime = 0

mpDraw = mp.solutions.drawing_utils #method to draw at points

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                if id == 4 or id == 8 or id == 12 or id == 16 or id == 20:
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # draws the points and connections

######################################## fps calculations
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
########################################

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3) # displays fps

    cv2.imshow("Image", img) # displays original image
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
#########################################