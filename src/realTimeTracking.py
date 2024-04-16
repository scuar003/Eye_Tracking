import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist

def eye_aspect_ratio(eye):
    print(f"Eye Points: {eye}")  # Debugging: Check the eye points structure
    eye = np.array(eye)  # Convert list to numpy array for easier indexing

    p1, p2 = tuple(eye[1]), tuple(eye[5])
    p3, p4 = tuple(eye[2]), tuple(eye[4])
    p5, p6 = tuple(eye[0]), tuple(eye[3])

    print(f"Points: {p1}, {p2}, {p3}, {p4}, {p5}, {p6}")  # More debugging

    A = dist.euclidean(p1, p2)
    B = dist.euclidean(p3, p4)
    C = dist.euclidean(p5, p6)

    ear = (A + B) / (2.0 * C)
    return ear


# Initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("src/Images/shape_predictor_68_face_landmarks.dat")

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
        
        # Use the average EAR for both eyes for simplification
        ear = (leftEAR + rightEAR) / 2.0
        
        # Visualize the eye regions
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
