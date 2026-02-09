import cv2
from ultralytics import YOLO

# Загружаем модель (легкая, норм для CPU)
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Не удалось открыть камеру")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO ожидает BGR, OpenCV тоже — ок
    results = model(frame, verbose=False)

    h, w, _ = frame.shape
    frame_center = (w // 2, h // 2)

    # Рисуем центр кадра
    cv2.circle(frame, frame_center, 5, (255, 0, 0), -1)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if model.names[cls_id] != "person":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # центр человека
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    cv2.imshow("Person detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
