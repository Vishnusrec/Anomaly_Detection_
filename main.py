import cv2
import numpy as np
import time
import threading
import tflite_runtime.interpreter as tflite

MODEL_PATH = "model_int8.tflite"
INPUT_SIZE = 320
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.45

CLASSES = [
    "pothole",
    "crack",
    "speed_breaker",
    "debris",
    "waterlogging"
]

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = box1_area + box2_area - inter_area
    return inter_area / union if union > 0 else 0

def non_max_suppression(boxes, scores, iou_threshold):
    indices = np.argsort(scores)[::-1]
    selected = []

    while len(indices) > 0:
        current = indices[0]
        selected.append(current)
        rest = indices[1:]

        keep = []
        for idx in rest:
            if compute_iou(boxes[current], boxes[idx]) < iou_threshold:
                keep.append(idx)

        indices = np.array(keep)

    return selected

class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                return
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy()

    def stop(self):
        self.stopped = True
        self.cap.release()

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

vs = VideoStream(0).start()
time.sleep(1)

while True:
    ret, frame = vs.read()
    if not ret:
        break

    h, w, _ = frame.shape

    roi = frame[h//2:h, :]
    roi_resized = cv2.resize(roi, (INPUT_SIZE, INPUT_SIZE))

    input_data = np.expand_dims(roi_resized, axis=0).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    boxes = []
    scores = []
    class_ids = []

    for det in output_data:
        score = det[4]
        if score > CONF_THRESHOLD:
            x_center, y_center, bw, bh = det[0], det[1], det[2], det[3]

            x1 = int((x_center - bw/2) * w)
            y1 = int((y_center - bh/2) * h/2 + h//2)
            x2 = int((x_center + bw/2) * w)
            y2 = int((y_center + bh/2) * h/2 + h//2)

            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            class_ids.append(int(det[5]))

    if len(boxes) > 0:
        selected = non_max_suppression(boxes, scores, IOU_THRESHOLD)
        for i in selected:
            timestamp = int(time.time())
            cv2.imwrite(f"anomaly_{timestamp}.jpg", frame)
            break

vs.stop()
