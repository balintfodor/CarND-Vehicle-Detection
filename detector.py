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

    def _multi_scale(self, cropped):
        yx0 = np.array([0, 0])
        yx1 = np.array([0, 0])
        hw0 = np.array([64, 1280])
        hw1 = np.array([256, 1280])
        yx = lambda f : f * yx0 + (1 - f) * yx1
        hw = lambda f : f * hw0 + (1 - f) * hw1
        rois = []
        n = 2
        for i in range(n + 1):
            rois.append({'yx': yx(1/n*i), 'hw': hw(1/n*i)})

        r = []
        for d in rois:
            a = d['yx']
            b = d['hw']
            ys = int(a[0])
            ye = int(a[0] + b[0])
            xs = int(a[1])
            xe = int(a[1] + b[1])
            im = cropped[ys:ye, xs:xe]
            f = 64 / im.shape[0]
            im = resize(im, (64, int(1280 * f)), order=0)
            r.append((im, (ys, xs, f)))
        return r

    def _process_image(self, image, out_dir, idx):
        cropped = image[400:656, :, :]
        ms = self._multi_scale(cropped)
        out = cropped[:, :, :]
        for i, (im, orig) in enumerate(ms):
            # imsave('{}/{:02d}-{}-scale.png'.format(out_dir, idx, i), im)
            feat_map = self.trainer.preprocess(im)
            sws = self.trainer.search_window_size()
            win = self.sliding_window(feat_map, (sws[0], sws[1]), (sws[0] // 4, sws[1] // 4))
            im_out = np.copy(im)
            for (ys, ye), (xs, xe) in win:
                pred = self.trainer.classify(feat_map[ys:ye, xs:xe].ravel())
                xy_s = self.trainer.win_to_img((ys, xs))
                xy_e = self.trainer.win_to_img((ye, xe))
                p1 = (int(xy_s[1]), int(xy_s[0]))
                p2 = (int(xy_e[1]), int(xy_e[0]))
                if pred:
                    cv2.rectangle(im_out, p1, p2, (1, 0, 0))
                    ys, xs, f = orig
                    p1 = (int(xy_s[1] / f - ys), int(xy_s[0] / f - xs))
                    p2 = (int(xy_e[1] / f - ys), int(xy_e[0] / f - xs))
                    cv2.rectangle(out, p1, p2, (255, 0, 0))
                imsave('{}/{:02d}-{}-rect.png'.format(out_dir, idx, i), im_out)
        imsave('{}/{:02d}-all.png'.format(out_dir, idx), out)

        # im_out = np.copy(image)

        # sws = self.trainer.search_window_size()
        # win = self.sliding_window(hog_map_1, (sws[0], sws[1]), (sws[0] // 4, sws[1] // 4))
        # for (ys, ye), (xs, xe) in win:
        #     pred = self.trainer.classify(hog_map_1[ys:ye, xs:xe].ravel())
        #     xy_s = self.trainer.win_to_img((ys, xs))
        #     xy_e = self.trainer.win_to_img((ye, xe))
        #     p1 = (int(xy_s[1]), int(xy_s[0]))
        #     p2 = (int(xy_e[1]), int(xy_e[0]))
        #     if pred:
        #         cv2.rectangle(im_out, p1, p2, (255, 0, 0))

        # imsave('{}/{:02d}-rect.png'.format(out_dir, idx), im_out)

        # for hm, f in maps[:1]:
        #     win = self.sliding_window(hm, (bs[0], bs[1]), (bs[0] // 1, bs[1] // 1))
        #     out = np.zeros((hm.shape[0], hm.shape[1]))
        #     for (ys, ye), (xs, xe) in win:
        #         pred = self.trainer.classify(hm[ys:ye, xs:xe].ravel())
        #         out[ys:ye, xs:xe] += pred
        #         xy_s = self.trainer.hog_map_coord_to_img(ys, xs)
        #         xy_e = self.trainer.hog_map_coord_to_img(ye, xe)
        #         p1 = (int(xy_s[1]*f), int(xy_s[0]*f))
        #         p2 = (int(xy_e[1]*f), int(xy_e[0]*f))
        #         cv2.rectangle(im_out, p1, p2, (255, 0, 0))
        #         # out[int(ys * f):int(ye * f), int(xs * f):int(xe * f)] += pred

            # im_out[:, :, 1] += resize(out, image.shape[:2], order=0) * 0.01
            # out /= np.maximum(np.max(out), 1)
            # imsave('{}/{:02d}-pred-{}.png'.format(out_dir, idx, f), out)
        # im_out /= np.max(im_out)
        # imsave('{}/{:02d}-heat.png'.format(out_dir, idx), im_out)

        # out /= np.max(out)
        # out = resize(out, image.shape, order=0)
        # imsave('{}/{:02d}-pred.png'.format(out_dir, idx), out)
        # imsave('{}/{:02d}-orig.png'.format(out_dir, idx), image)