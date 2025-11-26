# main.py
# YOLOv8 Face Recognition (improved UI + temporal smoothing)
# Requirements: ultralytics, opencv-python, numpy
#
# Pastikan model:
# - yolov8n-face.pt                 (detector)
# - runs/classify/train/weights/best.pt   (recognizer)
#
# Jalankan:
# python main.py

from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import deque, defaultdict

# ---------------------------
# Configurable parameters
# ---------------------------
DETECTOR_WEIGHTS = "yolov8n-face.pt"
RECOGNIZER_WEIGHTS = "runs/classify/train/weights/best.pt"

DETECT_CONF_MIN = 0.30      # minimal confidence detection (detector)
RECOG_CONF_MIN = 0.60       # minimal confidence recognition to accept label
MIN_BOX_AREA = 1000         # minimal area (w*h) for a face box (filter tiny boxes)
SMOOTH_WINDOW = 7           # number of latest predictions to use for smoothing per track
IOU_MATCH_THRESHOLD = 0.3   # IoU threshold to match detections to existing tracks
TRACK_TIMEOUT = 1.5         # seconds to keep a track without updates

# UI settings
BOX_COLOR_GOOD = (0, 200, 0)
BOX_COLOR_LOW = (0, 180, 255)
TEXT_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ---------------------------
# Helper functions
# ---------------------------
def iou(boxA, boxB):
    # boxes: (x1,y1,x2,y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    boxAArea = max(1, (boxA[2]-boxA[0]) * (boxA[3]-boxA[1]))
    boxBArea = max(1, (boxB[2]-boxB[0]) * (boxB[3]-boxB[1]))
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def draw_rounded_rect(img, pt1, pt2, color, thickness=2, radius=8):
    # minimal rounded rect: draw rectangle + corner circles
    x1,y1 = pt1; x2,y2 = pt2
    # rectangle
    cv2.rectangle(img, (x1+radius, y1), (x2-radius, y2), color, thickness, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1+radius), (x2, y2-radius), color, thickness, cv2.LINE_AA)
    # corners
    cv2.ellipse(img, (x1+radius, y1+radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2-radius, y1+radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1+radius, y2-radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2-radius, y2-radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)

def draw_label(img, text, x, y, bg_color=(0,0,0,150), text_color=TEXT_COLOR):
    # draw semi-transparent rectangle + text
    font = FONT
    scale = 0.7
    thickness = 2
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 6
    x2 = x + w + pad*2
    y2 = y - h - pad
    if y2 < 0:
        y2 = 0
    overlay = img.copy()
    cv2.rectangle(overlay, (x, max(0,y - h - pad*2)), (x2, y + pad//2), (0,0,0), -1)
    alpha = 0.45
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, text, (x + pad, y - 6), font, scale, text_color, thickness, cv2.LINE_AA)

# ---------------------------
# Tracking helpers
# ---------------------------
class Track:
    def __init__(self, bbox, track_id, timestamp):
        self.bbox = bbox  # last bbox (x1,y1,x2,y2)
        self.id = track_id
        self.last_seen = timestamp
        self.history = deque(maxlen=SMOOTH_WINDOW)  # store recent predicted names (or None/'Unknown')
        self.scores = deque(maxlen=SMOOTH_WINDOW)   # store recent confidences

    def update(self, bbox, name, score, timestamp):
        self.bbox = bbox
        self.last_seen = timestamp
        self.history.append(name)
        self.scores.append(score)

    def get_display_name(self):
        # majority vote among history, require average confidence
        if len(self.history) == 0:
            return "Unknown", 0.0
        # count non-Unknown
        votes = defaultdict(int)
        sum_scores = defaultdict(float)
        for name, score in zip(self.history, self.scores):
            votes[name] += 1
            sum_scores[name] += score
        # choose top voted name
        top_name = max(votes.items(), key=lambda x: (x[1], sum_scores[x[0]]))[0]
        avg_score = sum_scores[top_name] / max(1, votes[top_name])
        # require at least 50% of window votes for a name (to avoid flicker)
        if votes[top_name] >= max(1, int(0.5 * self.history.maxlen)):
            return top_name, avg_score
        else:
            return "Unknown", avg_score

# ---------------------------
# Main
# ---------------------------

def main():
    # Load models
    try:
        detector = YOLO(DETECTOR_WEIGHTS)
    except Exception as e:
        print(f"[ERROR] gagal load detector weights '{DETECTOR_WEIGHTS}': {e}")
        return

    try:
        recognizer = YOLO(RECOGNIZER_WEIGHTS)
    except Exception as e:
        print(f"[ERROR] gagal load recognizer weights '{RECOGNIZER_WEIGHTS}': {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Kamera tidak terbuka.")
        return

    # optionally set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracks = dict()   # track_id -> Track
    next_track_id = 0

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        now = time.time()
        dt = now - prev_time if prev_time else 0.0
        prev_time = now
        fps = 1.0 / dt if dt > 0 else 0.0

        # DETECTION
        dets = detector(frame, conf=DETECT_CONF_MIN, imgsz=640)[0]  # returns list-like; take first
        detected_boxes = []
        detected_scores = []
        # parse boxes
        if hasattr(dets, "boxes") and len(dets.boxes) > 0:
            for b in dets.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy()) if hasattr(b.xyxy[0], "cpu") else map(int, b.xyxy[0])
                conf_det = float(b.conf[0]) if hasattr(b, "conf") else 1.0
                # area filter
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                area = w * h
                if area < MIN_BOX_AREA:
                    continue
                # clip to image
                h_img, w_img = frame.shape[:2]
                x1 = max(0, min(x1, w_img-1))
                x2 = max(0, min(x2, w_img-1))
                y1 = max(0, min(y1, h_img-1))
                y2 = max(0, min(y2, h_img-1))
                detected_boxes.append((x1,y1,x2,y2))
                detected_scores.append(conf_det)

        # match detections to existing tracks via IoU
        unmatched_dets = set(range(len(detected_boxes)))
        matched_tracks = set()

        iou_matrix = []
        # build iou matrix
        for t_id, tr in tracks.items():
            row = []
            for d_idx, db in enumerate(detected_boxes):
                row.append(iou(tr.bbox, db))
            iou_matrix.append((t_id, row))

        # greedy matching: for each track, find best detection > threshold
        for t_id, row in iou_matrix:
            best_idx = np.argmax(row) if len(row)>0 else -1
            best_iou = row[best_idx] if best_idx>=0 else 0
            if best_iou >= IOU_MATCH_THRESHOLD and best_idx in unmatched_dets:
                # assign
                db = detected_boxes[best_idx]
                x1,y1,x2,y2 = db
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue
                # recognition
                try:
                    r = recognizer(face_crop, imgsz=224, conf=0.0)[0]
                    pred_idx = int(r.probs.top1) if hasattr(r.probs, "top1") else int(r.probs.top1)
                    name = r.names[pred_idx]
                    conf = float(r.probs.top1conf) if hasattr(r.probs, "top1conf") else float(r.probs.top1conf)
                except Exception:
                    name = "Unknown"
                    conf = 0.0
                # apply recognition threshold
                if conf < RECOG_CONF_MIN:
                    name_disp = "Unknown"
                else:
                    name_disp = name
                tracks[t_id].update(db, name_disp, conf, now)
                unmatched_dets.discard(best_idx)
                matched_tracks.add(t_id)

        # remaining unmatched detections -> create new tracks
        for d_idx in list(unmatched_dets):
            db = detected_boxes[d_idx]
            x1,y1,x2,y2 = db
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            try:
                r = recognizer(face_crop, imgsz=224, conf=0.0)[0]
                pred_idx = int(r.probs.top1)
                name = r.names[pred_idx]
                conf = float(r.probs.top1conf)
            except Exception:
                name = "Unknown"
                conf = 0.0
            name_disp = name if conf >= RECOG_CONF_MIN else "Unknown"
            tr = Track(db, next_track_id, now)
            tr.update(db, name_disp, conf, now)
            tracks[next_track_id] = tr
            next_track_id += 1

        # remove stale tracks
        stale_ids = []
        for t_id, tr in tracks.items():
            if now - tr.last_seen > TRACK_TIMEOUT:
                stale_ids.append(t_id)
        for t in stale_ids:
            del tracks[t]

        # Draw UI
        overlay = frame.copy()
        for t_id, tr in tracks.items():
            x1,y1,x2,y2 = tr.bbox
            display_name, avg_score = tr.get_display_name()
            color = BOX_COLOR_GOOD if display_name != "Unknown" and avg_score >= RECOG_CONF_MIN else BOX_COLOR_LOW

            # rounded box
            draw_rounded_rect(overlay, (x1,y1), (x2,y2), color, thickness=2)

            # label with confidence
            label_text = f"{display_name} ({avg_score:.2f})" if display_name != "Unknown" else "Unknown"
            draw_label(overlay, label_text, x1, y1)

        # blend overlay for nicer UI
        alpha = 0.9
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # fps
        cv2.putText(frame, f"FPS: {fps:.1f}", (20,40), FONT, 1.0, (200,200,200), 2, cv2.LINE_AA)

        cv2.imshow("YOLOv8 Face Recognition (improved UI)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
