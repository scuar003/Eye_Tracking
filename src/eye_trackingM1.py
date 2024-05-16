import wx
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

class CameraApp(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Camera App", size=(400, 200))
        
        self.panel = wx.Panel(self)
        
        self.camera_label = wx.StaticText(self.panel, label="Select Camera:")
        self.camera_choice = wx.Choice(self.panel, choices=["Camera 1", "Camera 2"]) # Add more camera options as needed
        
        self.live_feed_button = wx.Button(self.panel, label="Start Eye-Tracking")
        self.live_feed_button.Bind(wx.EVT_BUTTON, self.start_live_feed)
        
        self.stop_feed_button = wx.Button(self.panel, label="Stop Eye-Tracking")
        self.stop_feed_button.Bind(wx.EVT_BUTTON, self.stop_live_feed)
        self.stop_feed_button.Disable() # Initially disabled until live feed is started
        
        self.layout = wx.BoxSizer(wx.VERTICAL)
        self.layout.Add(self.camera_label, 0, wx.ALL, 5)
        self.layout.Add(self.camera_choice, 0, wx.ALL | wx.EXPAND, 5)
        self.layout.Add(self.live_feed_button, 0, wx.ALL | wx.EXPAND, 5)
        self.layout.Add(self.stop_feed_button, 0, wx.ALL | wx.EXPAND, 5)
        
        self.panel.SetSizerAndFit(self.layout)

        self.camera = None
        self.is_running = False
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("src/Models/shape_predictor_68_face_landmarks.dat")
        
    def eye_aspect_ratio(self, eye):
        eye = np.array(eye)
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def midpoint(self, p1, p2):
        return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

    def get_gaze_ratio(self, eye_points, gray):
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
            gaze_ratio_horizontal = 1
        elif right_side_white == 0:
            gaze_ratio_horizontal = 5
        else:
            gaze_ratio_horizontal = left_side_white / right_side_white

        top_side_threshold = threshold_eye[0:int(height / 2), 0:width]
        top_side_white = cv2.countNonZero(top_side_threshold)

        bottom_side_threshold = threshold_eye[int(height / 2):height, 0:width]
        bottom_side_white = cv2.countNonZero(bottom_side_threshold)

        if top_side_white == 0:
            gaze_ratio_vertical = 1
        elif bottom_side_white == 0:
            gaze_ratio_vertical = 5
        else:
            gaze_ratio_vertical = top_side_white / bottom_side_white

        return gaze_ratio_horizontal, gaze_ratio_vertical

    def draw_gaze_circle(self, eye_points, gaze_ratio_horizontal, gaze_ratio_vertical, frame):
        if len(eye_points) != 6:
            print(f"Error: Unexpected number of eye points detected: {len(eye_points)}")
            return

        eye_center = self.midpoint(eye_points[0], eye_points[3])
        length = 15  # Adjust the length to be closer to the iris/pupil

        end_point = eye_center

        if gaze_ratio_horizontal <= 0.9:
            end_point = (eye_center[0] - length, end_point[1])
        elif gaze_ratio_horizontal > 1.1:
            end_point = (eye_center[0] + length, end_point[1])

        if gaze_ratio_vertical <= 0.9:
            end_point = (end_point[0], eye_center[1] - length)
        elif gaze_ratio_vertical > 1.1:
            end_point = (end_point[0], eye_center[1] + length)

        cv2.circle(frame, end_point, 5, (255, 255, 255), 2)
        
    def start_live_feed(self, event):
        # Open camera based on selected camera index
        selected_camera_index = self.camera_choice.GetSelection()
        self.camera = cv2.VideoCapture(selected_camera_index)
        
        if self.camera.isOpened():
            self.live_feed_button.Disable()
            self.stop_feed_button.Enable()
            self.is_running = True
            while self.is_running:
                ret, frame = self.camera.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = self.detector(gray, 0)

                gaze_direction = "CENTER, CENTER"  # Default value
                
                for rect in rects:
                    shape = self.predictor(gray, rect)
                    shape = np.array([[p.x, p.y] for p in shape.parts()])
                    
                    if shape.shape[0] != 68:
                        print(f"Error: Unexpected number of landmark points detected: {shape.shape[0]}")
                        continue
                    
                    left_eye = shape[42:48]
                    right_eye = shape[36:42]
                    
                    leftEAR = self.eye_aspect_ratio(left_eye)
                    rightEAR = self.eye_aspect_ratio(right_eye)
                    
                    ear = (leftEAR + rightEAR) / 2.0
                    
                    left_eye_hull = cv2.convexHull(left_eye)
                    right_eye_hull = cv2.convexHull(right_eye)
                    cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
                    
                    left_gaze_ratio_horizontal, left_gaze_ratio_vertical = self.get_gaze_ratio(left_eye, gray)
                    right_gaze_ratio_horizontal, right_gaze_ratio_vertical = self.get_gaze_ratio(right_eye, gray)
                    
                    gaze_ratio_horizontal = (left_gaze_ratio_horizontal + right_gaze_ratio_horizontal) / 2.0
                    gaze_ratio_vertical = (left_gaze_ratio_vertical + right_gaze_ratio_vertical) / 2.0
                    
                    if gaze_ratio_horizontal <= 0.9:
                        gaze_direction_horizontal = "RIGHT"
                    elif gaze_ratio_horizontal > 1.1:
                        gaze_direction_horizontal = "LEFT"
                    else:
                        gaze_direction_horizontal = "CENTER"

                    if gaze_ratio_vertical <= 0.9:
                        gaze_direction_vertical = "DOWN"
                    elif gaze_ratio_vertical > 1.1:
                        gaze_direction_vertical = "UP"
                    else:
                        gaze_direction_vertical = "CENTER"

                    gaze_direction = f"{gaze_direction_horizontal}, {gaze_direction_vertical}"
                    
                    self.draw_gaze_circle(left_eye, left_gaze_ratio_horizontal, left_gaze_ratio_vertical, frame)
                    self.draw_gaze_circle(right_eye, right_gaze_ratio_horizontal, right_gaze_ratio_vertical, frame)
                
                # Mirror the frame for visualization
                frame = cv2.flip(frame, 1)

                # Draw the text after flipping the frame
                cv2.putText(frame, f"Gaze: {gaze_direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            self.camera.release()
            cv2.destroyAllWindows()
            self.live_feed_button.Enable()
            self.stop_feed_button.Disable()
        else:
            wx.MessageBox("Failed to open selected camera!", "Error", wx.OK | wx.ICON_ERROR)
        
    def stop_live_feed(self, event):
        self.is_running = False
        if self.camera and self.camera.isOpened():
            self.camera.release()
            cv2.destroyAllWindows()
            self.live_feed_button.Enable()
            self.stop_feed_button.Disable()

if __name__ == "__main__":
    app = wx.App(False)
    frame = CameraApp()
    frame.Show(True)
    app.MainLoop()
