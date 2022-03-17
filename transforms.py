import numpy as np


class NormalizeIntensity:
    def __call__(self, img):
        img -= img.min()
        img /= img.max()
        img *= 2048
        img -= 1024
        return img


class XRayCenterCrop:
    def crop_center(self, img):
        y, x, _ = img.shape
        crop_size = np.min([y,x])
        startx = x // 2 - (crop_size // 2)
        starty = y // 2 - (crop_size // 2)
        return img[starty:starty + crop_size, startx:startx + crop_size, :]

    def __call__(self, img):
        return self.crop_center(img)

# class ToRGB:
#     def __call__(self, img):
#         import pdb
#         pdb.set_trace()
#         repeated_channels_img =  np.stack((img,)*3, axis=-1)
#         return repeated_channels_img