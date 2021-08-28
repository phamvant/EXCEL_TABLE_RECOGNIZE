import cv2
from PIL import Image, ImageEnhance
import numpy as np
import openpyxl
import pprint
import imutils

import openpyxl
from yaml import loader
from yaml.error import Mark

wb = openpyxl.Workbook()
ws = wb.active

sheet = wb['Sheet']
def preprocess(img, factor: int):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Sharpness(img).enhance(factor)
    if gray.std() < 30:
        enhancer = ImageEnhance.Contrast(enhancer).enhance(factor)
    return np.array(enhancer)


table_image = cv2.imread("hi(1).jpg")
#table_image = cv2.imread("table.png")
gray = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY)
thresh, img_bin = cv2.threshold(
    gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
img_bin = 255-img_bin

kernel_len = gray.shape[1]//120
hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
image_horizontal = cv2.erode(img_bin, hor_kernel, iterations=3)
horizontal_lines = cv2.dilate(image_horizontal, hor_kernel, iterations=3)

h_lines = cv2.HoughLinesP(
    horizontal_lines, 1, np.pi/180, 30, maxLineGap=250)

def Show(table_image):
    table_image = imutils.resize(table_image, width=1200)
    cv2.imshow('a', table_image)
    cv2.waitKey()

def group_h_lines(h_lines, thin_thresh):
    new_h_lines = []
    while len(h_lines) > 0:
        thresh = sorted(h_lines, key=lambda x: x[0][1])[0][0]
        lines = [line for line in h_lines if thresh[1] - thin_thresh <= line[0][1] <= thresh[1] + thin_thresh]
        h_lines = [line for line in h_lines if thresh[1] - thin_thresh > line[0][1] or line[0][1] > thresh[1] + thin_thresh]
        x = []
        for line in lines:
            x.append(line[0][0])
            x.append(line[0][2])
        x_min, x_max = min(x) - int(5*thin_thresh), max(x) + int(5*thin_thresh)
        new_h_lines.append([x_min, thresh[1], x_max, thresh[1]])
    return new_h_lines
    
new_horizontal_lines = group_h_lines(h_lines, kernel_len)

kernel_len = gray.shape[1]//120
ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
image_vertical = cv2.erode(img_bin, ver_kernel, iterations=3)
vertical_lines = cv2.dilate(image_vertical, ver_kernel, iterations=3)

v_lines = cv2.HoughLinesP(vertical_lines, 1, np.pi/180, 30, maxLineGap=250)

def group_v_lines(v_lines, thin_thresh):
    new_v_lines = []
    while len(v_lines) > 0:
        thresh = sorted(v_lines, key=lambda x: x[0][0])[0][0]
        lines = [line for line in v_lines if thresh[0] - thin_thresh <= line[0][0] <= thresh[0] + thin_thresh]
        v_lines = [line for line in v_lines if thresh[0] - thin_thresh > line[0][0] or line[0][0] > thresh[0] + thin_thresh]
        y = []
        for line in lines:
            y.append(line[0][1])
            y.append(line[0][3])
        y_min, y_max = min(y) - int(4*thin_thresh), max(y) + int(4*thin_thresh)
        new_v_lines.append([thresh[0], y_min, thresh[0], y_max])
    return new_v_lines
    
new_vertical_lines = group_v_lines(v_lines, kernel_len)

def seg_intersect(line1: list, line2: list):
    a1, a2 = line1
    b1, b2 = line2
    da = a2-a1
    db = b2-b1
    dp = a1-b1

    def perp(a):
        b = np.empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b

    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float))*db + b1

points = []
for hline in new_horizontal_lines:
    x1A, y1A, x2A, y2A = hline
    for vline in new_vertical_lines:
        x1B, y1B, x2B, y2B = vline

        line1 = [np.array([x1A, y1A]), np.array([x2A, y2A])]
        line2 = [np.array([x1B, y1B]), np.array([x2B, y2B])]

        x, y = seg_intersect(line1, line2)
        if x1A <= x <= x2A and y1B <= y <= y2B:
            points.append([int(x), int(y)])

def get_bottom_right(right_points, bottom_points, points):
    for right in right_points:
        for bottom in bottom_points:
            if [right[0], bottom[1]] in points:
                return right[0], bottom[1]
    return None, None


cells = []

for point in points:
    left, top = point
    right_points = sorted(
        [p for p in points if p[0] > left and p[1] == top], key=lambda x: x[0])
    bottom_points = sorted(
        [p for p in points if p[1] > top and p[0] == left], key=lambda x: x[1])

    right, bottom = get_bottom_right(
        right_points, bottom_points, points)
    if right and bottom:
        cv2.rectangle(table_image, (left, top), (right, bottom), (0, 0, 255), 2)
        cells.append([left, top, right, bottom])

coord_x = []
coord_y = []
kkk = 0
kkk2 = 0
while (kkk < table_image.shape[1]):
    if (table_image[400][kkk][0] == 0 and table_image[400][kkk][2] == 255):
        coord_x.append(kkk)
        kkk += 20
    kkk += 1

while (kkk2 < table_image.shape[0]):
    if (table_image[kkk2][400][0] == 0 and table_image[kkk2][400][2] == 255):
        coord_y.append(kkk2)
        kkk2 += 10
    kkk2 += 1

coord = []
for j in range(len(coord_y) - 1):
    for i in range(len(coord_x) - 1):
        coord.append((coord_x[i],coord_y[j],coord_x[i + 1], coord_y[j + 1]))

import torch
from craft_structure.detection import detect, get_detector

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
craft = get_detector("models/craft_mlt_25k.pth", device)

final_horizontal_list = []
horizontal_list1 = []

for cellll in cells:
    cell_x_min, cell_y_min, cell_x_max, cell_y_max = cellll
    cell_image = table_image[cell_y_min:cell_y_max, cell_x_min:cell_x_max]
    horizontal_list, free_list = detect(craft, cell_image, device=device)
    horizontal_list1.append(horizontal_list)
    for box in horizontal_list:
        x_min = cell_x_min + box[0]
        x_max = cell_x_min + box[1]
        y_min = cell_y_min + box[2]
        y_max = cell_y_min + box[3]
        cv2.rectangle(table_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        final_horizontal_list.append([x_min, x_max, y_min, y_max])

# cv2.imshow('a', table_image)
# cv2.waitKey()

from vietocr_structure.Predictor import Predictor
from vietocr_structure.vocab import Vocab
from vietocr_structure.load_config import Cfg
from vietocr_structure.ocr_model import VietOCR
from vietocr_structure.ocr_utils import get_image_list

def build_model(config):
    vocab = Vocab(config['vocab'])
    model = VietOCR(len(vocab),
                    config['backbone'],
                    config['cnn'],
                    config['transformer'],
                    config['seq_modeling'])

    model = model.to(config['device'])
    return model, vocab

# Load model ocr
config = Cfg.load_config_from_file('/home/thuan/Desktop/Desktop/vietocr_structure/config/vgg-seq2seq.yml')
config['predictor']['beamsearch'] = False
model, vocab = build_model(config)
model.load_state_dict(torch.load(
    'models/vgg-seq2seq.pth', map_location=config['device']), strict=False)

def ocr(img, textline_list, imgH=32):
    image_list, max_width = get_image_list(
        textline_list, img, model_height=imgH)

    coordinate_list = [x[0] for x in image_list]
    crop_img_list = [x[1] for x in image_list]

    # load model ocr
    ocr_model = Predictor(model=model, config=config, vocab=vocab)
    set_bucket_thresh = config['set_bucket_thresh']

    # predict
    ocr_result = ocr_model.batch_predict(crop_img_list, set_bucket_thresh)
    final_result = list(zip(coordinate_list, ocr_result))
    return final_result

table_result = ocr(img=table_image, textline_list=final_horizontal_list)

def center(point):
      midx = (point[0][0] + point[2][0]) // 2
      midy = (point[0][1] + point[2][1]) // 2
      return midx, midy
#[(36, 35, 132, 253), (36, 253, 132, 345), (36, 345, 132, 431)
i = 1
j = 1
for celll in range(len(coord)):
    if j == 12:
        i += 1
        j = 1
    cell_x_min, cell_y_min, cell_x_max, cell_y_max = coord[celll]
    final = ""
    space = " "
    for a in range(len(table_result)):
        mid_x = center(table_result[a][0])[0]
        mid_y = center(table_result[a][0])[1]
        if cell_x_min < mid_x < cell_x_max and cell_y_min < mid_y < cell_y_max:
            if table_result[a][1] + space == final:
                final += ""
            else:
                final += table_result[a][1] + space
    sheet.cell(row=i, column=j, value=final)
    j += 1

wb.save('result.xlsx')
cv2.imwrite("result.jpg", table_image)
# Show(table_image)