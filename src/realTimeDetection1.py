import cv2
import face_recognition

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("error could not opened video stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't retrieve frame from video stream")
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_frame = frame[:, :, ::-1]

    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

    for face_landmarks in face_landmarks_list:
        # Draw circles around the eyes
        for point in face_landmarks['left_eye']:
            cv2.circle(frame, point, 3, (255, 0, 0), -1)  # Blue color for left eye
        for point in face_landmarks['right_eye']:
            cv2.circle(frame, point, 3, (0, 255, 0), -1)  # Green color for right eye

    # frame = cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_DEFAULT)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close windows
cap.release()
cv2.destroyAllWindows()