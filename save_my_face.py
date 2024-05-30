import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
face_detection = mp_face_detection.FaceDetection()


def save_face_image(frame, bbox):
    x, y, w, h = bbox
    face_image = frame[y:y + h, x:x + w]
    cv2.imwrite("face_image.jpg", face_image)


cap = cv2.VideoCapture(0)

face_saved = True

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face = face_detection.process(frame_rgb)

    if results_face.detections and face_saved:
        for detection in results_face.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            save_face_image(frame, (x, y, w, h))
            face_saved = False

    cv2.imshow('Frame', frame)
    if cv2.waitKey(10) & 0xFF == ord('c'):
        break

cap.release()
cv2.destroyAllWindows()
