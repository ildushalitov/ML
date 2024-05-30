import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
face_detection = mp_face_detection.FaceDetection()
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)


def is_my_face(face_image):
    if face_image is None or face_image.size == 0:
        return False
    sample_face_image = cv2.imread("face_image.jpg")
    sample_face_gray = cv2.cvtColor(sample_face_image, cv2.COLOR_BGR2GRAY)
    face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    correlation = cv2.compareHist(cv2.calcHist([sample_face_gray], [0], None, [256], [0, 256]),
                                  cv2.calcHist([face_gray], [0], None, [256], [0, 256]),
                                  cv2.HISTCMP_CORREL)
    return correlation > 0.67


def determine_name_fingers(num_fingers, is_my_face):
    if is_my_face:
        if num_fingers == 1:
            return "Ildus"
        elif num_fingers == 2:
            return "Halitov"
    return ""


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face = face_detection.process(frame_rgb)

    if results_face and results_face.detections:
        for detection in results_face.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            face_image = frame[y:y + h, x:x + w]
            if is_my_face(face_image):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                results_hands = hands.process(frame_rgb)
                num_fingers = 0

                if results_hands.multi_hand_landmarks:
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        thumb_tip = hand_landmarks.landmark[4]
                        index_tip = hand_landmarks.landmark[8]
                        middle_tip = hand_landmarks.landmark[12]
                        ring_tip = hand_landmarks.landmark[16]
                        pinky_tip = hand_landmarks.landmark[20]

                        fingers_up = [
                            thumb_tip.x < hand_landmarks.landmark[3].x,
                            index_tip.y < hand_landmarks.landmark[5].y,
                            middle_tip.y < hand_landmarks.landmark[9].y,
                            ring_tip.y < hand_landmarks.landmark[13].y,
                            pinky_tip.y < hand_landmarks.landmark[17].y
                        ]

                        num_fingers += fingers_up.count(True)

                name_fingers = determine_name_fingers(num_fingers - 1, True)
                cv2.putText(
                    frame,
                    name_fingers,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    'Unknown',
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )

    cv2.imshow('Frame', frame)
    if cv2.waitKey(10) & 0xFF == ord('c'):
        break

cap.release()
cv2.destroyAllWindows()
