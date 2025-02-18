import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n-pose.pt")

# Open video file
video_path = "side_squad.mp4"
cap = cv2.VideoCapture(video_path)
#for real time video
#cap=cv2.VideoCapture(0)

# calculate angle
def calculate_angle(a, b, c):
    """ hip (a), knee (b), ankle (c)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    
    ab = np.linalg.norm(a - b)
    bc = np.linalg.norm(b - c)
    ac = np.linalg.norm(a - c)

    if 2 * ab * bc == 0:
        return 180  

    angle = np.degrees(np.arccos((ab**2 + bc**2 - ac**2) / (2 * ab * bc)))
    return angle
squat_count = 0
squat_in_progress = False
tracker = True

while cap.isOpened():

    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 360)) 

    if not ret:
        break 

    results = model.track(frame, persist=True)  
    for result in results:
        if result.boxes is not None and len(result.boxes.xyxy) > 0:
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)                        
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)      # Draw rectangle around detected persons

        if result.keypoints is not None and len(result.keypoints.data) > 0:
            kpts = result.keypoints.data.cpu().numpy()[0]  # Take first detected person
            if len(kpts) >= 17:
                # Extract keypoints
                left_hip, right_hip = kpts[11][:2], kpts[12][:2]
                left_knee, right_knee = kpts[13][:2], kpts[14][:2]
                left_ankle, right_ankle = kpts[15][:2], kpts[16][:2]

                #for Drawing keypoints on frame
                for point in [left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]:
                    cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 255, 255), -1)

                # Connect keypoints with lines
                lines = [(left_hip, left_knee), (left_knee, left_ankle),
                         (right_hip, right_knee), (right_knee, right_ankle)]
                for p1, p2 in lines:
                    cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 255), 2)

                # Calculate knee angles
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

                
                cv2.putText(frame, f"Left Angle: {int(left_knee_angle)}", # Display left knee angles
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 150, 90), 2)
                cv2.putText(frame, f"Right Angle: {int(right_knee_angle)}", # Display right knee angles
                            (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 150, 24), 2)

                if left_knee_angle < 90 and right_knee_angle < 90:
                     squat_in_progress = True
                elif squat_in_progress and left_knee_angle > 150 and right_knee_angle > 150:  
                     squat_count += 1
                     squat_in_progress = False

    # Display squat count
    cv2.putText(frame, f"Squats: {squat_count}", 
                (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the video 
    cv2.imshow("Squat Tracker", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()                           #For releasing resources

cv2.destroyAllWindows()