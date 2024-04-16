import cv2
import face_recognition
from PIL import Image, ImageDraw  # Import the required classes from PIL

# Load your image or video
image = face_recognition.load_image_file("src/Images/face1.jpg")

# Find all facial features in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(image)

for face_landmarks in face_landmarks_list:
    # Print the location of each eye
    print("The eyes are located at the following coordinates:")
    print(face_landmarks['left_eye'])
    print(face_landmarks['right_eye'])

    # Show the image with eyes highlighted
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)

    d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(255, 255, 255), width=5)
    d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(255, 255, 255), width=5)

    pil_image.show()
