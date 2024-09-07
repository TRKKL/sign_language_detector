import cv2
import mediapipe as mp
import time
import numpy as np


# Activate Mediapipe hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

target_time = 1
pre_shape = None
time_start = None

# Determining mouse click positions.
def mouse_click(event, x, y, flags, param):
    global queue
    if event == cv2.EVENT_LBUTTONDOWN:
        if width-120 < x < width-20 and 20 < y < 70:  # JPEG  button
            img = cv2.imread('info.jpg')
            if img is not None:
                cv2.imshow('JPEG Image', img)   
        elif width-120 < x < width-20 and 90 < y < 140:  # Reset button
            queue = ["System", "Active:"]
        elif width-120 < x < width-20 and 160 < y < 210:  # quit button
            cv2.destroyAllWindows()
            exit()


def stable_time_check(queue,current_shape):

    global pre_shape, time_start

    #this part give us time to change hand shape before second scan.
    if current_shape != pre_shape:
        pre_shape = current_shape
        time_start = time.time()

    #we add queue 
    elif time.time() - time_start >= target_time:
        queue.append(current_shape)
        print(queue[-1], end=" ")
        pre_shape = None  

#determine queue
queue = []
queue.append("System")
queue.append("Active:")
threshold = 0.9 


cap = cv2.VideoCapture(0)
cv2.namedWindow("Cam", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Cam", mouse_click)




while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converting from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Hand shape detection
    result = hands.process(rgb_frame)


    # Let's add space to see our queue
    height, width, _ = frame.shape
    black_bar = np.zeros((100, width, 3), dtype=np.uint8)
    black_bar[:] = (0, 0, 0)

    #this part we design out Buttons 
    cv2.rectangle(frame, (width-20, 20), (width-120, 70), (0, 255, 0), -1)
    cv2.putText(frame, "Information", (width-110, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.rectangle(frame, (width-20, 90), (width-120, 140), (0, 0, 255), -1)
    cv2.putText(frame, "Reset", (width-100, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.rectangle(frame, (width-20, 160), (width-120, 210), (255, 0, 0), -1)
    cv2.putText(frame, "Quit", (width-100, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    if result.multi_hand_landmarks:
        
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            hand_label = result.multi_handedness[idx].classification[0].label
            #mcp: joint where palm and finger meet
            # tip: top op the finger
            #.x and .y is show us their plain. x axes and y axes
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
            pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
            

            if (
                # we wrote  hand_label == "Right" but because of the reflection we have to see left hand
                hand_label == "Right" and
                thumb_tip.x < middle_tip.x and
                thumb_tip.x > index_tip.x and
                index_tip.y > index_mcp.y and
                middle_tip.y > middle_mcp.y and
                ring_tip.y > ring_mcp.y and
                pinky_tip.y > pinky_mcp.y
            ):
                
                stable_time_check(queue,"T")

            elif (
                hand_label == "Right" and
                pinky_tip.y > pinky_mcp.y and
                thumb_tip.y > thumb_mcp.y and
                pinky_tip.y >  index_mcp.y and
                pinky_tip.y > middle_mcp.y and
                pinky_tip.y > ring_mcp.y 
                
            ):
                
                stable_time_check(queue,"J")

            elif (
         
         
                hand_label == "Left" and
                thumb_tip.y < middle_mcp.y and
                thumb_tip.x < ring_tip.x and  
                index_tip.x < index_mcp.x and  
                middle_tip.x < middle_mcp.x and  
                ring_tip.x > ring_mcp.x and  
                pinky_tip.x > pinky_mcp.x  
            ):
                
                stable_time_check(queue,"H")


            elif (
                hand_label == "Left" and
                index_tip.y < index_mcp.y and
                middle_tip.y < middle_mcp.y and
                ring_tip.y > ring_mcp.y and
                pinky_tip.y > pinky_mcp.y and
                thumb_tip.x > index_tip.x and
                thumb_tip.x > middle_tip.x  
            ):
                
                stable_time_check(queue,"R")

            elif (
                hand_label == "Left" and 
                index_tip.y < index_mcp.y and
                middle_tip.y < middle_mcp.y and
                ring_tip.y > ring_mcp.y and
                pinky_tip.y > pinky_mcp.y and
                thumb_tip.x < index_tip.x
            ):
                
                stable_time_check(queue,"V")
            
            elif (
                hand_label == "Right" and
                thumb_tip.x < index_tip.x and 
                index_tip.y > index_mcp.y and 
                middle_tip.y > middle_mcp.y and
                ring_tip.y > ring_mcp.y and
                pinky_tip.y > pinky_mcp.y
                
            ):
                
                stable_time_check(queue,"A")


                
            elif (
                hand_label == "Left" and
                pinky_tip.x < ring_tip.x and
                ring_tip.x < middle_tip.x and
                index_tip.y < index_mcp.y and
                middle_tip.y < middle_mcp.y and
                ring_tip.y < ring_mcp.y and
                pinky_tip.y < pinky_mcp.y and
                thumb_tip.x < index_tip.x 
                   
                      
            ):
                
                stable_time_check(queue,"B")

            elif (
                hand_label == "Left" and
                thumb_tip.y > index_mcp.y and
                index_tip.y < index_mcp.y and
                middle_tip.y < middle_mcp.y and
                ring_tip.y < ring_mcp.y and
                pinky_tip.y < pinky_mcp.y 
                   
                   
            ):
               
                stable_time_check(queue,"C")
                

            elif (
                hand_label == "Right" and
                thumb_tip.y > index_tip.y and
                thumb_tip.y > middle_tip.y and
                thumb_tip.y > ring_tip.y and
                thumb_tip.y > pinky_tip.y and
                index_tip.y > index_mcp.y and
                middle_tip.y > middle_mcp.y and
                ring_tip.y > ring_mcp.y and
                pinky_tip.y > pinky_mcp.y
            ):
                
                stable_time_check(queue,"E")


            elif (
                hand_label == "Right" and
                index_tip.y < index_mcp.y and
                thumb_tip.x > index_tip.x and
                middle_tip.y > middle_mcp.y and
                ring_tip.y > ring_mcp.y and
                pinky_tip.y > pinky_mcp.y
            ):
                
                stable_time_check(queue,"D")


            elif (
                hand_label == "Right" and
                index_tip.y < index_mcp.y and
                thumb_tip.x < index_tip.x and
                middle_tip.y > middle_mcp.y and
                ring_tip.y > ring_mcp.y and
                pinky_tip.y > pinky_mcp.y
            ):
                (queue, "L")
                stable_time_check(queue,"L")


            elif (
                hand_label == "Left" and
                thumb_tip.y < middle_mcp.y and
                index_tip.x < index_mcp.x and 
                middle_tip.x > middle_mcp.x and  
                ring_tip.x > ring_mcp.x and  
                pinky_tip.x > pinky_mcp.x   
                
            ):
              
                stable_time_check(queue,"G")


            elif (
                hand_label == "Right" and
                pinky_tip.y < pinky_mcp.y and
                thumb_tip.x < index_tip.x and
                index_tip.y > index_mcp.y and
                middle_tip.y > middle_mcp.y and
                ring_tip.y > ring_mcp.y
            ):
               
                stable_time_check(queue,"I")


            elif (
                hand_label == "Right" and
                thumb_tip.y < thumb_mcp.y and
                index_tip.y < index_mcp.y and
                middle_tip.y < middle_mcp.y and
                ring_tip.y > ring_mcp.y and
                pinky_tip.y > pinky_mcp.y and
                thumb_tip.x > index_tip.x and  
                thumb_tip.x < middle_tip.x 
            ):
                
                stable_time_check(queue,"K")



            elif (
                hand_label == "Left" and
                thumb_tip.x > ring_tip.x and  
                thumb_tip.x < middle_tip.x and 
                index_tip.y > index_mcp.y and
                middle_tip.y > middle_mcp.y and
                ring_tip.y > ring_mcp.y and
                pinky_tip.y > pinky_mcp.y
            ):
                
                stable_time_check(queue, "N")

            elif (
                hand_label == "Right" and
                thumb_tip.x < pinky_tip.x and
                thumb_tip.x > middle_tip.x and
                index_tip.y > index_mcp.y and
                middle_tip.y > middle_mcp.y and
                ring_tip.y > ring_mcp.y and
                pinky_tip.y > pinky_mcp.y
            ):
                
                stable_time_check(queue, "M") 

            elif (
                hand_label == "Left" and
                index_tip.y < index_mcp.y and 
                index_tip.x > index_mcp.x and
                middle_tip.y > middle_mcp.y and 
                ring_tip.y > ring_mcp.y and  
                pinky_tip.y > pinky_mcp.y 
            ):
                
                stable_time_check(queue, "X")

            
            elif (
                hand_label == "Left" and              
                thumb_tip.x < index_tip.x and  # Thumb to the left of (or below) the index finger
                thumb_tip.y < index_tip.y and  # Thumb under index finger
                middle_tip.y > middle_mcp.y and # Middle finger in palm (closed)
                ring_tip.y > ring_mcp.y and    # Ring finger in palm (closed)
                pinky_tip.y > pinky_mcp.y      # pinky finger in palm (closed)
            ):
                
                stable_time_check(queue,"P")

            elif (

                hand_label == "Left" and  
                thumb_tip.x > thumb_mcp.x and
                index_tip.x > index_mcp.x and
                middle_tip.y > middle_mcp.y and
                ring_tip.y > ring_mcp.y and
                pinky_tip.y > pinky_mcp.y 
            
            ):
                
                stable_time_check(queue,"Q")

            elif (
                hand_label == "Left" and
                thumb_tip.x < index_tip.x and 
                index_tip.y > index_mcp.y and 
                middle_tip.y > middle_mcp.y and
                ring_tip.y > ring_mcp.y and
                pinky_tip.y > pinky_mcp.y
            ):
               
                stable_time_check(queue,"S")

            elif (
                hand_label == "Left" and
                thumb_tip.x > middle_mcp.x and
                index_tip.x < thumb_tip.x and
                middle_tip.y > middle_mcp.y and
                ring_tip.y > ring_mcp.y and
                pinky_tip.y > pinky_mcp.y 
            ):
                
                stable_time_check(queue,"Z")

            elif (
                hand_label == "Left" and
            
                thumb_tip.y < thumb_mcp.y and
                index_tip.x > index_mcp.x and
                middle_tip.x  > middle_mcp.x and
                ring_tip.x > ring_mcp.x and
                pinky_tip.x  > pinky_mcp.x
           
            ):
             
                stable_time_check(queue, "O")


            elif (
                hand_label == "Right" and
                index_tip.y < index_mcp.y and 
                middle_tip.y < middle_mcp.y and  
                ring_tip.y > ring_mcp.y and  
                pinky_tip.y > pinky_mcp.y  
            ):
               
                stable_time_check(queue,"U")


            elif (
                hand_label == "Right" and
                index_tip.y < index_mcp.y and
                middle_tip.y < middle_mcp.y and
                ring_tip.y < ring_mcp.y and
                pinky_tip.y > pinky_mcp.y
            ):
               
                stable_time_check(queue,"W")


            elif (
                hand_label == "Left" and
                thumb_tip.y < thumb_mcp.y and  
                pinky_tip.y < pinky_mcp.y and  
                index_tip.y > index_mcp.y and 
                middle_tip.y > middle_mcp.y and  
                ring_tip.y > ring_mcp.y 
            ):
               
                stable_time_check(queue,"Y")

        
            elif (
                # abs: close enough
                hand_label == "Right" and
                abs(thumb_tip.x - index_tip.x) < threshold and # Thumb and index fingertips are close enough horizontally
                abs(thumb_tip.y - index_tip.y) < threshold and # Thumb and index fingertips are close enough vertically
                middle_tip.y < middle_mcp.y and # Middle finger is bent (tip below MCP joint)
                ring_tip.y < ring_mcp.y and # Ring finger is bent (tip below MCP joint)
                pinky_tip.y < pinky_mcp.y   # Pinky finger is bent (tip below MCP joint)
            ):
              
                stable_time_check(queue,"F")
                   

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

     # write letters to scrren
    if queue:
        cv2.putText(black_bar, " ".join(queue), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # add bar to the scrren
    frame = np.vstack((frame, black_bar))
    
    cv2.imshow("Cam", frame)

    # press "space" wo close the program
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break


cap.release()
cv2.destroyAllWindows()
