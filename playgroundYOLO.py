from ultralytics import YOLO
import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")


model = YOLO('yolov8x')
# model.predict('clips/0a2d9b_0.mp4',save=True)
results= model.predict('clips/0a2d9b_0.mp4', save=True, device=mps_device)
print(results[0])
print('--------------------------------')
for box in results[0].boxes:
    print(box)

