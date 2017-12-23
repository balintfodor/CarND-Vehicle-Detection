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
        # print(im.shape, win_size, win_step)
        # print(win_list)
        # exit(0)
        return win_list

    def process(self, out_dir):
        for idx, image in tqdm(enumerate(self.reader.next_frame()), total=self.reader.total):
            self._process_image(image, out_dir, idx)

    def _process_image(self, image, out_dir, idx):
        image = image[400:656, :, :]
        bs = (self.trainer.sample_block_h, self.trainer.sample_block_w)
        
        hog_map_1 = self.trainer.preprocess(image)
        hog_map_1 = hog_map_1.reshape((hog_map_1.shape[0], hog_map_1.shape[1], -1))

        maps = [(hog_map_1, 1)]
        for f in [1.5, 2, 3, 4]:
            maps.append((rescale(hog_map_1, 1/f, order=0), f))

        im_out = np.copy(image)

        for hm, f in maps[:1]:
            win = self.sliding_window(hm, (bs[0], bs[1]), (bs[0] // 1, bs[1] // 1))
            out = np.zeros((hm.shape[0], hm.shape[1]))
            for (ys, ye), (xs, xe) in win:
                pred = self.trainer.classify(hm[ys:ye, xs:xe].ravel())
                out[ys:ye, xs:xe] += pred
                xy_s = self.trainer.hog_map_coord_to_img(ys, xs)
                xy_e = self.trainer.hog_map_coord_to_img(ye, xe)
                p1 = (int(xy_s[1]*f), int(xy_s[0]*f))
                p2 = (int(xy_e[1]*f), int(xy_e[0]*f))
                cv2.rectangle(im_out, p1, p2, (255, 0, 0))
                # out[int(ys * f):int(ye * f), int(xs * f):int(xe * f)] += pred

            # im_out[:, :, 1] += resize(out, image.shape[:2], order=0) * 0.01
            # out /= np.maximum(np.max(out), 1)
            # imsave('{}/{:02d}-pred-{}.png'.format(out_dir, idx, f), out)
        # im_out /= np.max(im_out)
        imsave('{}/{:02d}-heat.png'.format(out_dir, idx), im_out)

        # out /= np.max(out)
        # out = resize(out, image.shape, order=0)
        # imsave('{}/{:02d}-pred.png'.format(out_dir, idx), out)
        # imsave('{}/{:02d}-orig.png'.format(out_dir, idx), image)