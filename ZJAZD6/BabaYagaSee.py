import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        _, image = cap.read()

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        left_eye_pos = results.pose_landmarks.landmark[1]
        right_eye_pos = results.pose_landmarks.landmark[4]
        between_eyes_x = ((left_eye_pos.x + right_eye_pos.x) / 2) * image.shape[1]
        between_eyes_y = ((left_eye_pos.y + right_eye_pos.y) / 2) * image.shape[0]

        surrender = False
        left_wrist_y = results.pose_landmarks.landmark[15].y * image.shape[0]
        right_wrist_y = results.pose_landmarks.landmark[16].y * image.shape[0]
        if left_wrist_y <=between_eyes_y and right_wrist_y <= between_eyes_y:
            surrender = True


        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if not surrender:
            cv2.circle(image, (int(between_eyes_x), int(between_eyes_y)), 15, (0, 255, 255), thickness= 4, lineType=cv2.FILLED)

        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        if cv2.waitKey(5) == 27:
            break
cap.release()