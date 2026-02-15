from PIL import Image
from torchvision import transforms

from application.common.data_preprocessing import PadToSquare

IMAGE_SQUARE_SIZE =256

def preprocess_img(img_path):
    with Image.open(img_path).convert("RGB") as img:
        width, height = img.size

        max_side = max(width, height)
        scale = 256 / max_side

        new_width = int(width * scale)
        new_height = int(height * scale)

        img_resized = img.resize((new_width, new_height), Image.LANCZOS)

        transform = transforms.Compose([
            PadToSquare(IMAGE_SQUARE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

        return transform(img_resized)