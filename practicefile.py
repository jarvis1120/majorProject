import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Pose class
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Drawing utility to display the landmarks on the image
mp_drawing = mp.solutions.drawing_utils

# OpenCV to capture webcam feed
cap = cv2.VideoCapture(0)

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    return abs(angle) if abs(angle) < 180 else 360 - abs(angle)

# Thresholds for squat detection
squat_threshold_down = 100  # Knee angle less than this indicates a squat down
squat_threshold_up = 160  # Knee angle greater than this indicates standing up
squat_count = 0
squat_position = 'up'  # Start in 'up' position
squat_status = 'Stand'  # Initial status

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Unable to receive frame.")
        break

    # Convert the BGR frame to RGB (MediaPipe uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for pose detection
    results = pose.process(rgb_frame)

    # Check if pose landmarks are detected
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get landmarks for the left leg (can do the same for right leg)
        landmarks = results.pose_landmarks.landmark

        # Extract relevant landmarks for squat detection (left hip, knee, ankle)
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]

        # Convert normalized coordinates to image size
        height, width, _ = frame.shape
        left_hip = [int(left_hip[0] * width), int(left_hip[1] * height)]
        left_knee = [int(left_knee[0] * width), int(left_knee[1] * height)]
        left_ankle = [int(left_ankle[0] * width), int(left_ankle[1] * height)]

        # Calculate knee angle using the three points
        knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

        # Debug output in terminal to check if knee angle is calculated
        print(f'Knee Angle: {int(knee_angle)}')

        # Display the calculated knee angle on the frame
        cv2.putText(frame, f'Knee Angle: {int(knee_angle)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Squat detection logic
        if knee_angle < squat_threshold_down and squat_position == 'up':
            squat_position = 'down'
            squat_status = 'Squat Down'
            print("Squat down detected")

        if knee_angle > squat_threshold_up and squat_position == 'down':
            squat_position = 'up'
            squat_count += 1
            squat_status = 'Squat Up'
            print("Squat up detected")
            print(f'Total Squats: {squat_count}')

        # Display squat count and status on the frame
        cv2.putText(frame, f'Squat Count: {squat_count}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, squat_status, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    else:
        # If no landmarks are detected, output in terminal for debug
        print("No landmarks detected!")

    # Display the processed frame
    cv2.imshow('BlazePose Squat Detection', frame)

    # Break loop with 'q' key
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
