import wx
import cv2

class CameraApp(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Camera App", size=(400, 200))
        
        self.panel = wx.Panel(self)
        
        self.camera_label = wx.StaticText(self.panel, label="Select Camera:")
        self.camera_choice = wx.Choice(self.panel, choices=["Camera 1", "Camera 2"]) # Add more camera options as needed
        
        self.live_feed_button = wx.Button(self.panel, label="Start Live Feed")
        self.live_feed_button.Bind(wx.EVT_BUTTON, self.start_live_feed)
        
        self.stop_feed_button = wx.Button(self.panel, label="Stop Live Feed")
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
                if ret:
                    # Display frame in OpenCV window (for demonstration)
                    cv2.imshow('Live Feed', frame)
                    # Check for quit command
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
            # Release the camera when done
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
