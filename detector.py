import numpy as np
from tqdm import tqdm

from skimage.io import imsave
from skimage.transform import pyramid_reduce, downscale_local_mean, rescale, resize
from skimage.exposure import rescale_intensity, equalize_adapthist
from skimage.color import rgb2hsv
import cv2


def uint_im(im):
    return (im * 255).astype(np.uint8)

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
        feat_map, yuv_map = self.trainer.image_to_hog_map(image)
        sws = self.trainer.search_window_size()
        win = self.sliding_window(feat_map, (sws[0], sws[1]), (1, 1))
        rois = []
        for (ys, ye), (xs, xe) in win:
            xy_s = self.trainer.win_to_img((ys - 1, xs - 1))
            xy_e = self.trainer.win_to_img((ye, xe))

            yuv_patch = yuv_map[xy_s[0]:xy_e[0], xy_s[1]:xy_e[1]]
            feat_patch = feat_map[ys:ye, xs:xe]

            vec = self.trainer.image_to_vector(yuv_patch, feat_patch)
            pred = self.trainer.classify(vec)
            
            if pred > 0.6:
                rois.append((xy_s, xy_e, pred[0]))
        return rois

    def _coord_to_tuple(self, s, e):
        return (int(s[1]), int(s[0])), (int(e[1]), int(e[0]))

    def _process_image(self, image, out_dir, idx):
        cropped = image[368:656, :, :]

        outs = []
        roi_list = []
        for scale in np.linspace(1, 4, 8):
            scaled = rescale(cropped, 1 / scale, order=1, mode='constant')
            rois = self._detect(scaled)
            for s, e, pred in rois:
                p1, p2 = self._coord_to_tuple(s, e)
                cv2.rectangle(scaled, p1, 
                    (int(p2[0] * pred + p1[0] * (1-pred)), int(p1[1] + 8)),
                    (1 - pred, pred, 0), -1)
                roi_param = np.array([s[0], s[1], e[0], e[1], pred]).ravel()
                roi_param[:4] *= scale
                roi_list.append(roi_param)
                cv2.rectangle(scaled, p1, p2, (1, 1, 0), 1)
            # imsave('{}/{:02d}-{}.png'.format(out_dir, idx, scale), uint_im(scaled))
            back_scaled = resize(scaled, cropped.shape, mode='constant', order=0)
            outs.append(back_scaled)

        if len(outs) > 0:
            outs = np.array(outs)
            out = np.mean(outs, axis=0)
            # imsave('{}/{:02d}-accum.png'.format(out_dir, idx), uint_im(out))
        else:
            # imsave('{}/{:02d}-accum.png'.format(out_dir, idx), uint_im(cropped))
            pass

        final_rois = self._select_final_rois(roi_list)
        for roi in final_rois:
            p1, p2 = self._coord_to_tuple(roi[0:2], roi[2:4])
            cv2.rectangle(cropped, p1, p2, (127, 255, 0), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(cropped, '{:.4f}'.format(roi[4]), (p1[0]+2, p2[1]-2), font, 0.33, (255, 0, 0))
        imsave('{}/{:02d}.png'.format(out_dir, idx), cropped)

    def _select_final_rois(self, roi_list, overlap_th=1, min_sum_score=0.5):
        boxes = np.array(roi_list)

         # code based on: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
        if boxes.shape[0] == 0:
            return []
    
        pick = []
    
        x1 = boxes[:, 1]
        y1 = boxes[:, 0]
        x2 = boxes[:, 3]
        y2 = boxes[:, 2]
    
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
    
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
    
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
    
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
    
            overlap = (w * h) / area[idxs[:last]]
            above_th = overlap > overlap_th
            if np.any(above_th):
                above_idxs = idxs[:last][above_th]
                boxes[i, 4] = np.sum(boxes[above_idxs, 4])
                for k in range(4):
                    boxes[i, k] = np.mean(boxes[above_idxs, k])

            wh = np.where(above_th)[0]
            idxs = np.delete(idxs, np.concatenate(([last], wh)))
    
        finals = boxes[pick]
        return finals[np.where(finals[:, 4] > min_sum_score)]