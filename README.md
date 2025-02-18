# Squat Tracker with YOLOv11n Pose Estimation
This project provides a real-time squat tracking system using YOLOv11n pose estimation and OpenCV. It detects the user’s body keypoints (hip, knee, and ankle) and tracks knee angles to count the number of squats performed. The system displays the squat count and the angle of the knees for each squat.

# Key Features:

. Pose Estimation: Uses YOLOv11n to detect keypoints on the human body (hip, knee, ankle).

. Real-Time Squat Tracking: Counts the number of squats based on the knee angle between 90° (downward) and 150° (upward).

. Knee Angle Calculation: Calculates the knee angle using the law of cosines to determine squat depth.

. Angle Display: Displays real-time knee angles for left and right legs.

. Performance Optimized: Resizes frames and uses YOLO’s tracking features for improved performance in real-time.

# How It Works:

. Keypoint Detection: The YOLOv11n model detects keypoints of the body (hip, knee, ankle) in each frame.

. Knee Angle Calculation: The angle at the knee joint is calculated using the detected keypoints.

. Squat Detection: The system tracks the knee angles and identifies when the user is performing a squat (angle < 90°) and when the squat is  completed (angle > 150°).

. Real-Time Feedback: Squat count is updated and displayed, along with the knee angle for each leg.
