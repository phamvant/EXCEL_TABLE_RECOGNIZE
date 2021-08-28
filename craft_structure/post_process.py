import numpy as np
import math
import cv2

def diff(input_list):
    return max(input_list)-min(input_list)


def group_text_box(polys, num_box=4, slope_ths = 0.1, ycenter_ths = 0.4, height_ths = 0.4, width_ths = 1.0, add_margin = 0.1, sort_output = True):
    # poly top-left, top-right, low-right, low-left
    horizontal_list, free_list, combined_list, merged_list = [],[],[],[]

    for poly in polys:
        x_max = max([poly[0],poly[2],poly[4],poly[6]])
        x_min = min([poly[0],poly[2],poly[4],poly[6]])
        y_max = max([poly[1],poly[3],poly[5],poly[7]])
        y_min = min([poly[1],poly[3],poly[5],poly[7]])
        horizontal_list.append([x_min, x_max, y_min, y_max, 0.5*(y_min+y_max), y_max-y_min])
    if sort_output:
        horizontal_list = sorted(horizontal_list, key=lambda item: item[4])

    combined_list = combine_box_line(horizontal_list, height_ths, ycenter_ths)

    list_padded = padding_height(combined_list)
    # merge list use sort again
    merged_list = merge_box_inline(
        list_padded, width_ths, add_margin, num_box)
    line_list = merge_box_inline(
        list_padded, width_ths, add_margin, num_box=99)
    iou_ths = 0.3
    final_list = combine_box_overlap(merged_list, iou_ths, add_margin)
    final_line_list = combine_box_overlap(line_list, iou_ths, add_margin)
    # combine box

    return final_list, final_line_list



def calculate_ios(box1, box2):
    # Calculate overlap area
    """
        this function to calbulate intersection over area of small box
    """
    dx = min(box1[1], box2[1]) - max(box1[0], box2[0]) + 1

    if dx < 0:
        return 0.0

    dy = min(box1[3], box2[3]) - max(box1[2], box2[2]) + 1

    if dy < 0:
        return 0.0

    overlap_area = dx * dy

    min_area = min((box1[1] - box1[0] + 1) * (box1[3] - box1[2] + 1),
                   (box2[1] - box2[0] + 1) * (box2[3] - box2[2] + 1))

    return overlap_area / min_area


def combine_box_line(horizontal_list, height_ths, ycenter_ths):
    """     
        purpose: return a list of list contains boxes that same line.
        new_box: list contains boxes in the same line, reset in new line
        combined_list: list contains list of new_box
        b_height: list contains height of boxes in new_box
        b_ycenter: list contains center height of boxes in new_box
        height_ths : threshold 
        satisfy condition when:
            absolute of (mean of b_height) minus (height of recent box) smaller than height_ths * (mean of b_height)
            absolute of (mean of b_ycenter) minus (center height of recent box) smaller than height_ths * (mean of b_ycenter)
    """
    new_box = []
    combined_list = []
    for poly in horizontal_list:
        if len(new_box) == 0:
            b_height = [poly[5]]
            b_ycenter = [poly[4]]
            new_box.append(poly)
        else:
            # comparable height and comparable y_center level up to ths* mean of height of box in new box
            if (abs(np.mean(b_ycenter) - poly[4]) < ycenter_ths*np.mean(b_height)):
                b_height.append(poly[5])
                b_ycenter.append(poly[4])
                new_box.append(poly)
            else:  # this
                b_height = [poly[5]]
                b_ycenter = [poly[4]]
                combined_list.append(new_box)
                new_box = [poly]
    combined_list.append(new_box)
    return combined_list


def merge_box_inline(combined_list, width_ths, add_margin, num_box):
    """
        purpose: return a list of list contains boxes that same line.
        combined_list: list contains list of new_box.

    """
    # merge list use sort again
    merged_list = []
    for boxes in combined_list:
        if len(boxes) == 1:  # one box per line
            box = boxes[0]
            margin = int(add_margin*box[5])
            merged_list.append(
                [box[0], box[1], box[2], box[3], (box[1]-box[0])*(box[3]-box[2])])
        else:  # multiple boxes per line
            boxes = sorted(boxes, key=lambda item: item[0])

            merged_box, new_box = [], []
            for box in boxes:
                if len(new_box) == 0:
                    x_max = box[1]
                    new_box.append(box)
                else:
                    # comparable distance between box[0] and x_max(x_max is max width of recently box) with threshold*height of current box
                    # merge boxes
                    if abs(box[0]-x_max) < width_ths*(box[3]-box[2]) and len(new_box) <= num_box:
                        x_max = box[1]
                        new_box.append(box)
                    else:
                        # if this condition is not satisfied, i create new box
                        x_max = box[1]
                        merged_box.append(new_box)
                        new_box = [box]
            if len(new_box) > 0:
                merged_box.append(new_box)
            # merge boxes to 1 box
            for mbox in merged_box:
                if len(mbox) != 1:  # adjacent box in same line
                    x_min = min(mbox, key=lambda x: x[0])[0]
                    x_max = max(mbox, key=lambda x: x[1])[1]
                    y_min = min(mbox, key=lambda x: x[2])[2]
                    y_max = max(mbox, key=lambda x: x[3])[3]

                    margin = int(add_margin*(y_max - y_min))

                    merged_list.append(
                        [x_min, x_max, y_min, y_max, (y_max-y_min+1)*(x_max-x_min+1)])
                else:  # non adjacent box in same line
                    box = mbox[0]
                    margin = int(add_margin*(box[3] - box[2]))
                    merged_list.append(
                        [box[0], box[1], box[2], box[3], (box[3]-box[2])*(box[1]-box[0])])
    return merged_list


def combine_box_overlap(merged_list, iou_ths, add_margin):
    """
        purpose: merge box that overlap
    """
    # merged_list = sorted(merged_list, key=lambda item: -item[4])
    combined_list_again = []
    new_box = []
    cnt = 0
    check = np.zeros(len(merged_list))
    for i in range(len(merged_list)):
        if check[i] == 1:
            continue
        new_box = []
        new_box.append(merged_list[i])
        check[i] = 1
        for j in range(i+1, len(merged_list)):
            if check[j] == 1:
                continue
            iou = calculate_ios(merged_list[i], merged_list[j])
            if(iou > iou_ths):
                new_box.append(merged_list[j])
                check[j] = 1
        combined_list_again.append(new_box)
    final_list = []
    for mbox in combined_list_again:
        if len(mbox) != 1:
            x_min = min(mbox, key=lambda x: x[0])[0]
            x_max = max(mbox, key=lambda x: x[1])[1]
            y_min = min(mbox, key=lambda x: x[2])[2]
            y_max = max(mbox, key=lambda x: x[3])[3]
            margin = int(add_margin*(y_max - y_min))
            final_list.append([x_min-margin, x_max+margin,
                               y_min-margin-5, y_max+margin])
        else:
            box = mbox[0]
            margin = int(add_margin*(box[3] - box[2]))
            final_list.append([box[0]-margin, box[1]+margin,
                               box[2]-margin-5, box[3]+margin])
    return final_list


def padding_height(combined_list):
    list_padding = []
    for line in combined_list:
        y_max = max(line, key=lambda x: x[3])[3]
        y_min = min(line, key=lambda x: x[2])[2]
        new_box = [[box[0], box[1], y_min, y_max, box[4], box[5]]
                   for box in line]
        list_padding.append(new_box)
    return list_padding
