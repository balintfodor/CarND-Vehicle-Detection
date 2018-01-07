import glob
import sys
import cv2
import os
from skimage.io import imread
import numpy as np

class FrameReader(object):
    def __init__(self, file):
        if os.path.isfile(file):
            self.video = cv2.VideoCapture(file)
            self.total = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.next_frame = self._next_video_frame
            self.is_video = True
        elif os.path.isdir(file):
            self.images = collect_images(file)
            self.total = len(self.images)
            self.next_frame = self._next_image
            self.is_video = False
        else:
            print('input error')
            sys.exit(2)

    def _next_video_frame(self):
        while(self.video.isOpened()):
            ret, frame = self.video.read()
            if frame is not None:
                a = np.stack((frame[:, :, 2], frame[:, :, 1], frame[:, :, 0]), axis=2)
                yield a
            else:
                break
        
    def _next_image(self):
        for f in self.images:
            yield imread(f)

def collect_images(folder):
    images = []
    for format in ['jpg', 'png', 'tif']:
        images.extend(glob.glob("{}/**/*.{}".format(folder, format), recursive=True))
    return images
