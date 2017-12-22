import numpy as np
from tqdm import tqdm

from skimage.io import imsave


class Detector(object):
    def __init__(self, trainer, reader, args):
        self.trainer = trainer
        self.reader = reader
        self.args = args

    def process(self, out_dir):
        for idx, image in tqdm(enumerate(self.reader.next_frame()), total=self.reader.total):
            self._process_image(image, out_dir, idx)

    def _process_image(self, image, out_dir, idx):
        image = image[400:656, :, :]
        imsave('{}/{:02d}.png'.format(out_dir, idx), image)
        hog_map = self.trainer.preprocess(image)
        
        wins = self.trainer.generate_rois(hog_map)
        out = self.trainer.classify(wins.reshape((-1, 576)))
        out = out.reshape((wins.shape[0], wins.shape[1]))

        # out = np.zeros((wins.shape[0], wins.shape[1]))
        # for i, roi_row in enumerate(wins):
        #     for j, roi in enumerate(roi_row):
        #         pred = self.trainer.classify(roi.ravel())
        #         out[i, j] = pred
        imsave('{}/{:02d}-pred.png'.format(out_dir, idx), out)