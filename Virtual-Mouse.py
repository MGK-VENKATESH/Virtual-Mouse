import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math

# Initialize Mediapipe and PyAutoGUI
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

cap = cv2.VideoCapture(0)

prev_click_time = 0
prev_scroll_time = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            if len(lmList) >= 13:
                index_x, index_y = lmList[8]
                middle_x, middle_y = lmList[12]
                thumb_x, thumb_y = lmList[4]

                # Draw circles on fingertips
                cv2.circle(img, (index_x, index_y), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (middle_x, middle_y), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (thumb_x, thumb_y), 10, (0, 255, 0), cv2.FILLED)

                # Move mouse using index finger
                screen_x = np.interp(index_x, (100, w - 100), (0, screen_width))
                screen_y = np.interp(index_y, (100, h - 100), (0, screen_height))
                pyautogui.moveTo(screen_x, screen_y)

                # Check for pinch (click) - distance between index and thumb
                pinch_dist = math.hypot(index_x - thumb_x, index_y - thumb_y)
                if pinch_dist < 30:
                    current_time = time.time()
                    if current_time - prev_click_time > 1:
                        pyautogui.click()
                        prev_click_time = current_time
                        cv2.circle(img, ((index_x + thumb_x) // 2, (index_y + thumb_y) // 2), 15, (0, 255, 255), cv2.FILLED)

                # Scroll with two fingers (index and middle)
                two_finger_dist = math.hypot(index_x - middle_x, index_y - middle_y)
                if two_finger_dist < 40:
                    vertical_movement = index_y - middle_y

                    if abs(vertical_movement) > 20:
                        current_time = time.time()
                        if current_time - prev_scroll_time > 0.5:
                            if vertical_movement > 0:
                                pyautogui.scroll(20)  # Scroll up
                            else:
                                pyautogui.scroll(-20)  # Scroll down
                            prev_scroll_time = current_time

    # Display the result
    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
