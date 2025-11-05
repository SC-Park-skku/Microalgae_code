import sys
import torch
import cv2
from pathlib import Path
import numpy as np
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.augmentations import letterbox

# === Custom ===
video_path = r"C:Video Path"
weights = r"C:Weight path"

# === Loading device, model ===
device = select_device('')
model = DetectMultiBackend(weights, device=device)
stride, names, pt = model.stride, model.names, model.pt
imgsz = 640  

# === Video load ===
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

input_dir, input_name = os.path.split(video_path)
name, ext = os.path.splitext(input_name)
output_path = os.path.join(input_dir, f"{name}_detected.mp4")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# === Frame ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("✅ End")
        break

    img_resized = letterbox(frame, imgsz, stride=stride, auto=True)[0]
    img = img_resized.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).to(device)
    img_tensor = img_tensor.float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)


    pred = model(img_tensor, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)


    for det in pred:
        if len(det):
            det = det.cpu().numpy()

            for *xyxy, conf, cls in det:
                cls = int(cls)
                label = f'{names[cls]} {conf:.2f}'

                # confidence
                if conf > 0.7:
                    color = (0, 255, 0)
                elif conf > 0.4:
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)

                # box, label
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 1)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save
    cv2.imshow("YOLOv5 Cell Detection", frame)
    out.write(frame)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

# Terminal
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Save complete: {output_path}")
