from selenium import webdriver
import json
import time
import keyboard
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

class TRex():

    def __init__(self,data):
        """Parameter initialization"""

        self.driver = webdriver.Chrome(data['driver_path'])

    def open_game(self):
        """Go to Google.com"""

        self.driver.get("https://www.google.com")

    def eye_aspect_ratio(self,eye):
        """Computes the EAR"""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])

        ear = (A + B) / (2 * C)
        return ear

    def play(self):
        """Play game"""

        RIGHT_EYE_POINTS = list(range(36, 42))
        LEFT_EYE_POINTS = list(range(42, 48))

        EYE_AR_THRESH = 0.22
        EYE_AR_CONSEC_FRAMES = 3
        EAR_AVG = 0

        # detection of the facial region
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


        COUNTER = 0
        TOTAL = 0

        # OpenCV - live video frame
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read() 
            if ret:
                # convert the frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 0)
                for rect in rects:
                    x = rect.left()
                    y = rect.top()
                    x1 = rect.right()
                    y1 = rect.bottom()
                    
                    landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()])
                    
                    left_eye = landmarks[LEFT_EYE_POINTS]
                    right_eye = landmarks[RIGHT_EYE_POINTS]
                    
                    left_eye_hull = cv2.convexHull(left_eye)
                    right_eye_hull = cv2.convexHull(right_eye)

                    ear_left = self.eye_aspect_ratio(left_eye)
                    ear_right = self.eye_aspect_ratio(right_eye)

                    ear_avg = (ear_left + ear_right) / 2.0

                    # detect of the blink
                    if ear_avg < EYE_AR_THRESH:
                        COUNTER += 1
                    else:
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                            TOTAL += 1
                            # press space bar when blinked
                            keyboard.press_and_release('space')
                            print("Eye blinked")
                        COUNTER = 0

                    cv2.putText(frame, "Blinks{}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1)
    
                cv2.imshow("Winks Found", frame)
                key = cv2.waitKey(1) & 0xFF

                if key is ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":

    with open('config.json') as config_file:
        data = json.load(config_file)

game = TRex(data)
game.open_game()
time.sleep(1)
game.play()