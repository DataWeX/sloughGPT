import sys
print("1")
sys.stdout.flush()
from domains.multimodal.vision import VisionCNN
print("2")
sys.stdout.flush()
m = VisionCNN()
print("3")
sys.stdout.flush()
m.build_model()
print("4")
sys.stdout.flush()
from PIL import Image
img = Image.new('RGB', (100, 100), color='blue')
cap = m.caption(img)
print(f'Caption: {cap.text}')
print("DONE")
