from detector import Detector
from config import Config
import os
import cv2
import numpy as np
# load model and dataset
state_file = "/home/lrq/tiny object detection 相关论文代码/SFD/dataset/logs_voc/models/epoch_662.pth.tar"
detector = Detector(state_file)

test_dir = Config.Test_yuncong_DIR
merged_dir = os.path.join(test_dir, 'merged_list.txt')
image_dir = os.path.join(test_dir, 'yuncong_data_new/Data/test/')
imgnames = open(merged_dir, 'r')
imgs = imgnames.readlines()
imgnames.close()

save_path = '/home/lrq/tiny object detection 相关论文代码/SFD/dataset/logs_voc/'

f = open(save_path + '/' + 'test_result_int.txt', 'w')
for img in range(len(imgs)):
    image = '%s/%s.jpg' % (image_dir, str(imgs[img][:-1]))
    bboxes = detector.infer(image)
    f.write('{:s}\n'.format('%s' % (imgs[img][:-1])))
    if bboxes is None:
        f.write('{:d}\n'.format(0))
        continue
    f.write('{:d}\n'.format(len(bboxes)))
    for b in bboxes:
        img2 = cv2.imread(image)
        sp = img2.shape
        x1 = int(max(0, b[1]))
        y1 = int(max(0, b[0]))
        x2 = int(min(b[3], sp[1]))
        y2 = int(min(b[2], sp[0]))
        s = b[4]
        f.write('{:d} {:d} {:d} {:d} {:.3f}\n'.format(x1, y1, (x2-x1+1), (y2-y1+1), s))