from vietocr_structure.translate import translate
import cv2
import numpy as np
import math
import torch
from collections import defaultdict

class Predictor(object):
    def __init__(self, model, config, vocab):
        self.config = config
        self.model = model
        self.vocab = vocab

    def predict(self, img):
        """
        Recognize image on batch size = 1

        Parameters:
        image: has shape of (H, W, C)

        Return:
        result(string): ocr result
        """

        img = self.preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        img = torch.FloatTensor(img)
        img = img.to(self.config['device'])

        s = translate(img, self.model)[0].tolist()
        s = self.vocab.decode(s)

        return s

    def batch_predict(self, images, set_bucket_thresh):
        """
        Recognize images on batch

        Parameters:
        images(list): list of cropped images
        set_buck_thresh(int): threshold to merge bucket in images

        Return:
        result(list string): ocr results
        """

        batch_dict, indices = self.batch_process(images, set_bucket_thresh)
        list_keys = [i for i in batch_dict if batch_dict[i]
                     != batch_dict.default_factory()]
        result = list([])

        for width in list_keys:
            batch = batch_dict[width]
            batch = np.asarray(batch)
            batch = torch.FloatTensor(batch)
            batch = batch.to(self.config['device'])
            sent = translate(batch, self.model).tolist()

            batch_text = self.vocab.batch_decode(sent)
            result.extend(batch_text)

        # sort text result corresponding to original coordinate
        z = zip(result, indices)
        sorted_result = sorted(z, key=lambda x: x[1])
        result, _ = zip(*sorted_result)

        return result

    def preprocess_input(self, image):
        """
        Preprocess input image (resize, normalize)

        Parameters:
        image: has shape of (H, W, C)

        Return:
        img: has shape (H, W, C)
        """

        h, w, _ = image.shape
        new_w, image_height = self.resize_v1(w, h, self.config['dataset']['image_height'],
                                             self.config['dataset']['image_min_width'],
                                             self.config['dataset']['image_max_width'])
        # cv2.imshow('a', image)
        # cv2.waitKey()
        img = cv2.resize(image, (new_w, image_height))
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))

        return img

    def batch_process(self, images, set_bucket_thresh):
        """
        Preprocess list input images and divide list input images to sub bucket which has same length 

        Parameters:
        image: has shape of (B, H, W, C)
            set_buck_thresh(int): threshold to merge bucket in images

        Return:
        batch_img_dict: list
            list of batch imgs
        indices: 
            position of each img in "images" argument
        """

        batch_img_dict = defaultdict(list)
        image_height = self.config['dataset']['image_height']

        img_li = [self.preprocess_input(img) for img in images]
        img_li, width_list, indices = self.sort_width(img_li, reverse=False)

        min_bucket_width = min(width_list)
        max_width = max(width_list)
        max_bucket_width = np.minimum(
            min_bucket_width + set_bucket_thresh, max_width)

        for i, image in enumerate(img_li):
            c, h, w = image.shape

            # reset min_bucket_width, max_bucket_width
            if w > max_bucket_width:
                min_bucket_width = w
                max_bucket_width = np.minimum(
                    min_bucket_width + set_bucket_thresh, max_width)

            avg_bucket_width = int((max_bucket_width + min_bucket_width) / 2)

            new_img = self.resize_v2(
                image, avg_bucket_width, height=image_height)
            batch_img_dict[avg_bucket_width].append(new_img)

        return batch_img_dict, indices

    @staticmethod
    def sort_width(batch_img: list, reverse: bool = False):
        """
        Sort list image correspondint to width of each image

        Parameters
        ----------
        batch_img: list
            list input image

        Return
        ------
        sorted_batch_img: list
            sorted input images
        width_img_list: list
            list of width images
        indices: list
            sorted position of each image in original batch images
        """
        def get_img_width(element):
            img = element[0]
            c, h, w = img.shape
            return w

        batch = list(zip(batch_img, range(len(batch_img))))
        sorted_batch = sorted(batch, key=get_img_width, reverse=reverse)
        sorted_batch_img, indices = list(zip(*sorted_batch))
        width_img_list = list(map(get_img_width, batch))

        return sorted_batch_img, width_img_list, indices

    @staticmethod
    def resize_v1(w: int, h: int, expected_height: int, image_min_width: int, image_max_width: int):
        """
        Get expected height and width of image

        Parameters
        ----------
        w: int
            width of image
        h: int
            height
        expected_height: int
        image_min_width: int
        image_max_width: int
            max_width of 

        Return
        ------

        """
        new_w = int(expected_height * float(w) / float(h))
        round_to = 10
        new_w = math.ceil(new_w / round_to) * round_to
        new_w = max(new_w, image_min_width)
        new_w = min(new_w, image_max_width)

        return new_w, expected_height

    @staticmethod
    def resize_v2(img, width, height):
        """
        Resize bucket images into fixed size to predict on  batch size
        """
        new_img = np.transpose(img, (1, 2, 0))
        new_img = cv2.resize(new_img, (width, height), cv2.INTER_AREA)
        new_img = np.transpose(new_img, (2, 0, 1))

        return new_img
