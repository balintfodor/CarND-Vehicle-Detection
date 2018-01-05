import numpy as np
from tqdm import tqdm

from skimage.io import imsave
from skimage.transform import pyramid_reduce, downscale_local_mean, rescale, resize
import cv2


class Detector(object):
    def __init__(self, trainer, reader, args):
        self.trainer = trainer
        self.reader = reader
        self.args = args

    def sliding_window(self, im, win_size, win_step):
        win_list = []
        for i in range(0, im.shape[0] - win_size[0] + 1, win_step[0]):
            for j in range(0, im.shape[1] - win_size[1] + 1, win_step[1]):
                win = ((i, i + win_size[0]), (j, j + win_size[1]))
                win_list.append(win)
        return win_list

    def process(self, out_dir):
        for idx, image in tqdm(enumerate(self.reader.next_frame()), total=self.reader.total):
            self._process_image(image, out_dir, idx)

    def _detect(self, image):
        feat_map = self.trainer.preprocess(image)
        sws = self.trainer.search_window_size()
        win = self.sliding_window(feat_map, (sws[0], sws[1]), (sws[0], sws[1]))
        rois = []
        for (ys, ye), (xs, xe) in win:
            pred = self.trainer.classify(feat_map[ys:ye, xs:xe].ravel())
            xy_s = self.trainer.win_to_img((ys, xs))
            xy_e = self.trainer.win_to_img((ye, xe))
            if pred > 0.5:
                rois.append((xy_s, xy_e, pred[0]))
        return rois

    def _coord_to_tuple(self, s, e):
        return ((s[1]), int(s[0])), (int(e[1]), int(e[0]))

    def _process_image(self, image, out_dir, idx):
        cropped = image[400:656, :, :]
        outs = []
        for scale in np.linspace(1, 4, 16):
            scaled = rescale(cropped, 1 / scale, order=0, mode='constant')
            rois = self._detect(scaled)
            for s, e, pred in rois:
                p1, p2 = self._coord_to_tuple(s, e)
                print(pred)
                cv2.rectangle(scaled, p1, p2, (1 - pred, pred, 0), -1)
                cv2.rectangle(scaled, p1, p2, (1, 1, 0), 2)
                back_scaled = resize(scaled, cropped.shape, mode='constant')
                outs.append(back_scaled)
        outs = np.array(outs)
        out = np.mean(outs, axis=0)
        imsave('{}/{:02d}.png'.format(out_dir, idx), out)
