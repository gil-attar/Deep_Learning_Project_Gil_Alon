"""
Debug script to visualize occlusion placement.
Shows original bbox vs occluded areas to check alignment.
"""

import json
import random
from pathlib import Path
from PIL import Image, ImageDraw

# Load test index
with open('data/processed/evaluation/test_index.json', 'r') as f:
    test_index = json.load(f)

# Pick first image with multiple objects
img_data = test_index['images'][0]
print(f"Image: {img_data['image_filename']}")
print(f"Objects: {img_data['num_objects']}")

# Load image
img_path = Path('data/raw/test/images') / img_data['image_filename']
if not img_path.exists():
    print(f"Image not found: {img_path}")
    exit(1)

image = Image.open(img_path).convert('RGB')
print(f"Image size: {image.size}")

# Draw original bboxes
img_with_boxes = image.copy()
draw = ImageDraw.Draw(img_with_boxes)

for box in img_data['ground_truth']:
    bbox = box['bbox_xyxy']
    print(f"\nClass: {box['class_name']}")
    print(f"  bbox_xyxy: {bbox}")

    # Draw green outline
    draw.rectangle(bbox, outline='lime', width=3)
    draw.text((bbox[0], bbox[1]-15), box['class_name'], fill='lime')

img_with_boxes.save('debug_original_boxes.jpg')
print("\n✓ Saved debug_original_boxes.jpg")

# Now apply 20% occlusion and show
from scripts.generate_synthetic_occlusions import apply_occlusions

occluded = apply_occlusions(
    image=image,
    boxes=img_data['ground_truth'],
    occlusion_level=0.2,
    base_seed=42
)

# Draw bboxes on occluded image
draw2 = ImageDraw.Draw(occluded)
for box in img_data['ground_truth']:
    bbox = box['bbox_xyxy']
    draw2.rectangle(bbox, outline='red', width=2)

occluded.save('debug_occluded.jpg')
print("✓ Saved debug_occluded.jpg")

print("\nCheck the images:")
print("  - Green boxes should align with actual ingredients")
print("  - Black occlusions should be INSIDE the red boxes")
print("  - If black is outside boxes, there's a coordinate mismatch")
