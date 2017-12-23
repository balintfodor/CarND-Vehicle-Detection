import numpy as np
from tqdm import tqdm

from skimage.io import imsave
from skimage.transform import pyramid_reduce, downscale_local_mean, rescale, resize


class Detector(object):
    def __init__(self, trainer, reader, args):
        self.trainer = trainer
        self.reader = reader
        self.args = args

    def sliding_window(self, im, win_size, win_step):
        win_list = []
        for i in range(0, im.shape[0] - win_size[0], win_step[0]):
            for j in range(0, im.shape[1] - win_size[1], win_step[1]):
                win = ((i, i + win_size[0]), (j, j + win_size[1]))
                win_list.append(win)
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
            imsave('{}/{:02d}-scale-{}.png'.format(out_dir, idx, f), rescale(image, 1/f))
            maps.append((rescale(hog_map_1, 1/f), f))

        out = np.zeros((hog_map_1.shape[0], hog_map_1.shape[1]))

        for hm, f in maps:
            win = self.sliding_window(hm, (bs[0], bs[1]), (1, 1))
            for (ys, ye), (xs, xe) in win:
                pred = self.trainer.classify(hm[ys:ye, xs:xe].ravel())
                out[int(ys * f):int(ye * f), int(xs * f):int(xe * f)] += pred

        out /= np.max(out)
        out = resize(out, image.shape, order=0)
        imsave('{}/{:02d}-pred.png'.format(out_dir, idx), out)
        imsave('{}/{:02d}-orig.png'.format(out_dir, idx), image)