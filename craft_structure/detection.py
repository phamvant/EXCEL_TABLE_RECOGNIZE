import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image
from collections import OrderedDict

import cv2
import numpy as np
from craft_structure.craft_utils import getDetBoxes, adjustResultCoordinates
from craft_structure.imgproc import resize_aspect_ratio, normalizeMeanVariance
from craft_structure.craft import CRAFT
from craft_structure.post_process import diff, group_text_box

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def test_net(canvas_size, mag_ratio, net, image, text_threshold, link_threshold, low_text, poly, device, estimate_num_chars=False):
    # resize
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, canvas_size,\
                                                                          interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    x = x.to(device)

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # Post-processing
    boxes, polys, mapper = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly, estimate_num_chars)

    # coordinate adjustment
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
    if estimate_num_chars:
        boxes = list(boxes)
        polys = list(polys)
    for k in range(len(polys)):
        if estimate_num_chars:
            boxes[k] = (boxes[k], mapper[k])
        if polys[k] is None: polys[k] = boxes[k]

    return boxes, polys

def get_detector(trained_model, device='cpu'):
    net = CRAFT()

    if device == 'cpu':
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location=device)))
    else:
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location=device)))
        net = torch.nn.DataParallel(net).to(device)
        cudnn.benchmark = False

    net.eval()
    return net

def get_textbox(detector, image, canvas_size, mag_ratio, text_threshold, link_threshold, low_text, poly, device, optimal_num_chars=None):
    result = []
    # cv2.imshow('a', image)
    # cv2.waitKey()

    estimate_num_chars = optimal_num_chars is not None
    bboxes, polys = test_net(canvas_size, mag_ratio, detector, image, text_threshold, link_threshold, low_text, poly, device, estimate_num_chars)
    if estimate_num_chars:
        polys = [p for p, _ in sorted(polys, key=lambda x: abs(optimal_num_chars - x[1]))]

    for i, box in enumerate(polys):
        poly = np.array(box).astype(np.int32).reshape((-1))
        result.append(poly)

    return result

def detect(detector, img, device, num_boxes=5, min_size = 10, text_threshold = 0.7, low_text = 0.4,
            link_threshold = 0.4,canvas_size = 2560, mag_ratio = 1.,
            slope_ths = 0.3, ycenter_ths = 0.3, height_ths = 0.7,
            width_ths = 1.0, add_margin = 0.1, reformat=True, optimal_num_chars=None):
    """
    output params:
    + horizontal_list: list of num words
    + line list: list of line
    
    """




    text_box = get_textbox(detector, img, canvas_size, mag_ratio,
                            text_threshold, link_threshold, low_text,
                            False, device, optimal_num_chars)
    if len(text_box) == 0:
        text_box.append(np.array([232,   7, 278,  13, 275,  33, 230,  27], dtype=np.int32))
    horizontal_list, line_list= group_text_box(text_box, num_boxes, slope_ths,
                                                ycenter_ths, height_ths,
                                                width_ths, add_margin, 
                                                (optimal_num_chars is None))
    print(line_list[0])
    print(horizontal_list[0])
    if min_size:
        horizontal_list = [i for i in horizontal_list if max(i[1]-i[0],i[3]-i[2]) > min_size]
        line_list = [i for i in line_list if max(i[1]-i[0],i[3]-i[2]) > min_size]

    return horizontal_list, line_list
