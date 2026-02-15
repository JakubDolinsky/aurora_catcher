import torchvision.transforms.functional as functional

class PadToSquare:
    def __init__(self, size = 256):
        self.size = size

    def __call__(self, img):
        w,h = img.size
        pad_w = self.size - w
        pad_h = self.size - h

        left =pad_w // 2
        right =pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        return functional.pad(
            img,
            [left, top, right, bottom],
            fill=0
        )