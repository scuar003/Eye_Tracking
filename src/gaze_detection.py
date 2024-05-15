import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist

def eye_aspect_ratio(eye):
    eye = np.array(eye)
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def midpoint(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

def get_gaze_ratio(eye_points, gray, frame):
    eye_region = np.array(eye_points, dtype=np.int32)
    mask = np.zeros_like(gray)
    cv2.fillConvexPoly(mask, eye_region, 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)
    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    eye = eye[min_y:max_y, min_x:max_x]
    _, threshold_eye = cv2.threshold(eye, 70, 255, cv2.THRESH_BINARY)
    threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
    height, width = threshold_eye.shape

    left_side_threshold = threshold_eye[0:height, 0:int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0:height, int(width / 2):width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("src/Models/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = np.matrix([[p.x, p.y] for p in shape.parts()])
        
        left_eye = shape[42:48]
        right_eye = shape[36:42]
        
        leftEAR = eye_aspect_ratio(left_eye)
        rightEAR = eye_aspect_ratio(right_eye)
        
        ear = (leftEAR + rightEAR) / 2.0
        
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
        
        left_gaze_ratio = get_gaze_ratio(left_eye, gray, frame)
        right_gaze_ratio = get_gaze_ratio(right_eye, gray, frame)
        
        gaze_ratio = (left_gaze_ratio + right_gaze_ratio) / 2.0
        
        if gaze_ratio <= 0.9:
            gaze = "RIGHT"
        elif gaze_ratio > 1.1:
            gaze = "LEFT"
        else:
            gaze = "CENTER"

        cv2.putText(frame, f"Gaze: {gaze}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
