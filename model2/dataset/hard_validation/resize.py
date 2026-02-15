import os
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(os.path.dirname(current_dir), "hard_validation_256")
os.makedirs(output_dir, exist_ok=True)

image_extensions = ('.jpg', '.jpeg')

for filename in os.listdir(current_dir):
    if filename.lower().endswith(image_extensions):
        input_path = os.path.join(current_dir, filename)
        output_path = os.path.join(output_dir, filename)

        with Image.open(input_path) as img:
            width, height = img.size

            max_side = max(width, height)
            scale = 256 / max_side

            new_width = int(width * scale)
            new_height = int(height * scale)

            img_resized = img.resize((new_width, new_height), Image.LANCZOS)
            img_resized.save(output_path)

print(f"\nDone. Images saved into: {output_dir}")
