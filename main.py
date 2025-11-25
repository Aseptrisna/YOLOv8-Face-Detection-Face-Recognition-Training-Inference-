from ultralytics import YOLO
import cv2

# YOLO untuk deteksi wajah
detector = YOLO("yolov8n-face.pt")

# YOLO untuk pengenalan wajah (hasil training)
recognizer = YOLO("runs/classify/train/weights/best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # DETEKSI WAJAH
    detect_results = detector(frame)[0]

    for box in detect_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        # RECOGNITION
        recog_result = recognizer(face_crop)[0]
        pred = recog_result.probs.top1
        name = recog_result.names[pred]
        conf = recog_result.probs.top1conf

        # TAMPILKAN
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{name} ({conf:.2f})", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("YOLOv8 Full Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
