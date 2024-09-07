import cv2
import numpy as np
import time
import mediapipe as mp
import handTrackingMod as htm

pTime = 0
pTime2 = 0
cTime = 0
cTime2 = 0

cap = cv2.VideoCapture(0) # image capture object

canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Function to draw lines on canvas
def draw_line(canvas, start, end, color, thickness=2):
    cv2.line(canvas, start, end, color, thickness)

# Function to erase drawn areas on canvas
def erase_area(canvas, center, radius, color):
    cv2.circle(canvas, center, radius, color, -1)

def export_canvas(canvas):
    f = open("imageExport.txt", "x")
    for col in canvas:
        for val in canvas:
            if(str(val[0]) != '0'):
                f.write("0 ")
            else:
                f.write(".")
        f.write('\n')
    
    f.close()

prev_x, prev_y = 0, 0

detector = htm.handDetection() # instance of the class handDetector

while True:
    success, img = cap.read()

    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    if len(lmList) != 0:
        check_draw = detector.checkDraw(lmList)
        check_erase = detector.checkErase(lmList)
        if check_draw:
                cv2.putText(img, "Detected", (10,150), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
                if prev_x != 0 and prev_y != 0:
                    draw_line(canvas, (prev_x, prev_y), (lmList[4][0], lmList[4][1]), (245, 152, 66))
                
        if check_erase and check_draw == False:
            if prev_x != 0 and prev_y != 0:
                    erase_area(canvas, (lmList[8][0], lmList[8][1]), 140, (0,0,0))

        prev_x, prev_y = lmList[4][0], lmList[4][1]
    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3) # displays fps

    cv2.imshow("Image", img) # displays original image
    cv2.imshow('Canvas', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# for col in canvas:
#     for val in canvas:
#         print(str(val) + " ")
#     print("\n")

export_canvas(canvas)