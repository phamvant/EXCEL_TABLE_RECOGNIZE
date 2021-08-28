import numpy as np
import math
import cv2
from skimage import io


def loadImage(img_file):
    img = io.imread(img_file)           # RGB order
    if img.shape[0] == 2:
        img = img[0]
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = np.array(img)
    return img


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


def remove_small_box(polys, small_ths):
    """
        purpose: remove box that have width and height bigger than small_ths
    """
    horizontal_list = []
    # remove small box with threshold
    for poly in polys:
        if poly[2] > small_ths and poly[3] > small_ths:
            x_max = poly[0] + poly[2]
            x_min = poly[0]
            y_max = poly[1] + poly[3]
            y_min = poly[1]
            width = x_max - x_min
            height = y_max - y_min
            horizontal_list.append(
                [x_min, x_max, y_min, y_max, 0.5*(y_min+y_max), y_max-y_min])
    horizontal_list = sorted(horizontal_list, key=lambda item: item[4])
    return horizontal_list


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


def group_text_box(polys, small_ths, num_box, iou_ths, ycenter_ths, height_ths, width_ths, add_margin):
    # poly top-left, top-right, low-right, low-left

    # remove small box with threshold
    horizontal_list = remove_small_box(polys, small_ths)
    # combine box
    # description of combined_list: each element of combined_list is a list contain boxes with same line
    combined_list = combine_box_line(horizontal_list, height_ths, ycenter_ths)

    list_padded = padding_height(combined_list)
    # merge list use sort again
    merged_list = merge_box_inline(
        list_padded, width_ths, add_margin, num_box)

    final_list = combine_box_overlap(merged_list, iou_ths, add_margin)

    return final_list


def four_point_transform(image, rect):
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1,
                                                maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def process_box(boxes):
    x_min = min([box["points"][0] for box in boxes])
    x_max = max([box["points"][1] for box in boxes])
    y_min = min([box["points"][2] for box in boxes])
    y_max = max([box["points"][3] for box in boxes])
    return [x_min, x_max, y_min, y_max]


def combine_box(horizontal_list, maximum_y, maximum_x, threshold_vertical=10, theshold_horizontal=50):
    new_list = []
    for box in horizontal_list:
        x_min = max(0, box[0])
        x_max = min(box[1], maximum_x)
        y_min = max(0, box[2])
        y_max = min(box[3], maximum_y)
        center_point = [x_min+(x_max-x_min)/2, y_min+(y_max-y_min)/2]
        new_list.append({
            "center_point": center_point,
            "distance": (x_max-x_min)/2,
            "points": [x_min, x_max, y_min, y_max]
        })
    p_b = []
    new_box = []
    for i in range(1, len(new_list)):
        if new_list[i]["points"][0] > new_list[i-1]["points"][0]:
            distance = new_list[i]["points"][0]-new_list[i-1]["points"][1]
        else:
            distance = new_list[i]["points"][1]-new_list[i-1]["points"][0]
        if abs(distance) < theshold_horizontal and abs(new_list[i]["center_point"][1]-new_list[i-1]["center_point"][1]) < threshold_vertical:
            p_b.append(new_list[i-1])
        elif len(p_b) > 0:
            p_b.append(new_list[i-1])
            new_box.append(process_box(p_b))
            p_b = []
        else:
            new_box.append(new_list[i-1]["points"])
            p_b = []
    new_box.append(new_list[-1]["points"])
    return new_box


def get_image_list(horizontal_list, img, model_height=32):
    image_list = []

    maximum_y, maximum_x, b = img.shape

    max_ratio_hori, max_ratio_free = 1, 1

    max_ratio_free = math.ceil(max_ratio_free)

    for idx, box in enumerate(horizontal_list):
        x_min = max(0, box[0])
        x_max = min(box[1], maximum_x)
        y_min = max(0, box[2])
        y_max = min(box[3], maximum_y)
        crop_img = img[y_min: y_max, x_min:x_max, :]
        width = x_max - x_min
        height = y_max - y_min
        ratio = width/height

        image_list.append(
            ([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], crop_img, idx))
        max_ratio_hori = max(ratio, max_ratio_hori)

    max_ratio_hori = math.ceil(max_ratio_hori)
    max_ratio = max(max_ratio_hori, max_ratio_free)
    max_width = math.ceil(max_ratio)*model_height

    # sort by vertical position
    # image_list = sorted(image_list, key=lambda item: item[0][0][1])

    return image_list, max_width
