import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

cap = cv2.VideoCapture(0)

prev_click_time = 0
prev_scroll_time = 0
scroll_direction = 0

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
                x1, y1 = lmList[8]
                x2, y2 = lmList[12]
                x3, y3 = lmList[4]

                cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x3, y3), 10, (0, 255, 0), cv2.FILLED)

                screen_x = np.interp(x1, (100, w - 100), (0, screen_width))
                screen_y = np.interp(y1, (100, h - 100), (0, screen_height))

                pyautogui.moveTo(screen_x, screen_y)

                dist = math.hypot(x2 - x1, y2 - y1)

                if dist < 30:
                    current_time = time.time()
                    if current_time - prev_click_time > 1:
                        pyautogui.click()
                        prev_click_time = current_time
                        cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 15, (0, 255, 255), cv2.FILLED)
                
                elif dist > 50:
                    vertical_move = y1 - y2

                    if abs(vertical_move) > 20:
                        current_time = time.time()
                        if current_time - prev_scroll_time > 0.5:
                            if vertical_move > 0:
                                pyautogui.scroll(10)
                            else:
                                pyautogui.scroll(-10)
                            prev_scroll_time = current_time

    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
