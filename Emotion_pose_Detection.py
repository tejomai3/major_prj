import cv2
import mediapipe as mp
import math

# Initialize MediaPipe solutions
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_holistic = mp.solutions.holistic

pose = mp_pose.Pose()
face_mesh = mp_face_mesh.FaceMesh()
holistic = mp_holistic.Holistic()

# Thresholds
DISTRESS_THRESHOLD = 3  # Number of frames to confirm distress
MALE_HEIGHT_RATIO = 0.8  # Heuristic for detecting male body ratio (shoulders wider)
FACE_LANDMARKS = {
    "left_eyebrow": [70, 63, 105],  # Approximate indices for eyebrows
    "right_eyebrow": [336, 296, 334],
    "mouth": [13, 14]  # Upper and lower lip
}

# Global variables
distress_count = 0
surrounding_male_count = 0

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def calculate_angle(p1, p2, p3):
    """Calculate the angle between three points."""
    angle = math.degrees(math.atan2(p3.y - p2.y, p3.x - p2.x) -
                         math.atan2(p1.y - p2.y, p1.x - p2.x))
    return abs(angle)

def is_hand_near_face(pose_landmarks):
    """Check if hand (wrist) is near the face (nose)."""
    nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

    left_hand_distance = calculate_distance(left_wrist, nose)
    right_hand_distance = calculate_distance(right_wrist, nose)

    return left_hand_distance < 0.1 or right_hand_distance < 0.1

def is_arm_raised(pose_landmarks):
    """Check if arm is raised based on elbow angle."""
    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    right_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    return left_arm_angle < 45 or right_arm_angle < 45

def detect_fear_or_distress(face_landmarks):
    """Detect fear or distress using facial landmarks."""
    if not face_landmarks:
        return False

    left_eyebrow = [face_landmarks.landmark[i] for i in FACE_LANDMARKS["left_eyebrow"]]
    right_eyebrow = [face_landmarks.landmark[i] for i in FACE_LANDMARKS["right_eyebrow"]]
    mouth = [face_landmarks.landmark[i] for i in FACE_LANDMARKS["mouth"]]

    # Calculate eyebrow height (y-coordinate difference)
    left_brow_height = abs(left_eyebrow[1].y - left_eyebrow[0].y)
    right_brow_height = abs(right_eyebrow[1].y - right_eyebrow[0].y)

    # Calculate mouth openness (y-distance between upper and lower lip)
    mouth_opening = abs(mouth[0].y - mouth[1].y)

    # Fear/distress is detected if eyebrows are raised and mouth is slightly open
    return left_brow_height > 0.02 and right_brow_height > 0.02 and mouth_opening > 0.015

def detect_surrounding_males(results):
    """Detects if a female is surrounded by males based on body structure heuristics."""
    global surrounding_male_count
    surrounding_male_count = 0

    for person in results.pose_landmarks:
        if person:
            left_shoulder = person.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = person.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = person.landmark[mp_pose.PoseLandmark.LEFT_HIP]

            # Calculate shoulder width relative to height
            shoulder_width = abs(left_shoulder.x - right_shoulder.x)
            body_height = abs(left_shoulder.y - left_hip.y)

            body_ratio = shoulder_width / body_height

            # If body ratio matches male characteristics, count it
            if body_ratio > MALE_HEIGHT_RATIO:
                surrounding_male_count += 1

    return surrounding_male_count > 1  # True if more than one male detected

def detect_threat(pose_landmarks, face_landmarks, results):
    """Detects threat based on pose, emotions, and surrounding males."""
    global distress_count

    # Conditions: Woman shows fear/distress & men are behind her
    if detect_fear_or_distress(face_landmarks) and detect_surrounding_males(results):
        distress_count += 1
    else:
        distress_count = max(0, distress_count - 1)  # Reduce count if condition not met

    # If distress persists for multiple frames, trigger alert
    if distress_count >= DISTRESS_THRESHOLD:
        print("🚨 ALERT: Potential Threat Detected! 🚨")
        distress_count = 0  # Reset after alert

    return distress_count >= DISTRESS_THRESHOLD
