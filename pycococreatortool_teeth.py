#!/usr/bin/env python3

import re
import datetime
import numpy as np
from itertools import groupby
from skimage import measure
from PIL import Image
from pycocotools import mask
import json
import os
from skimage.color import rgb2gray, gray2rgb
from skimage.io import imread, imshow, imsave
import cv2

INFO = {
    'description': 'TEETH DATASET',
    'url' : 'https://github.com/gil.../',
    'version': '0.1.0',
    'year': 2018,
    'contributor': 'IvisionLab',
    'date_created': datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [{
    'id': 1,
    'name': 'Attribution-NonCommercial-ShareAlike License',
    'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/'
}]

CATEGORIES = [{"supercategory": "teeth","id": 1,"name": "tooth"}]

convert = lambda text: int(text) if text.isdigit() else text.lower()
natrual_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]

def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
                counts.append(0)
        counts.append(len(list(elements)))

    return rle

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation

    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation 
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def get_teeth_masks(img_fullpath):
    image = imread(img_fullpath)
    image_gray = rgb2gray(image)
    image = gray2rgb(image)
    ret, thresh = cv2.threshold(image_gray, 0, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    annotations = []
    for id_contourn in range(len(contours)):
        img_cont = np.zeros(image.shape, dtype=image.dtype)
        cv2.drawContours(img_cont, contours, id_contourn, (255,255,255), -1)
        bimask = np.zeros(img_cont.shape[0:2], dtype=np.uint8)
        bimask = img_cont[:,:, 0]
        bimask = Image.fromarray(bimask)
        bimask = np.asarray(bimask.convert('1')).astype(np.uint8)
        annotations.append(bimask)
    return annotations


def get_teeth_masks_test(img_fullpath):
    image = imread(img_fullpath)
    image_gray = rgb2gray(image)
    image = gray2rgb(image)
    ret, thresh = cv2.threshold(image_gray, 0, 255, 0)
    
    annotations = []
    
    bimask = np.zeros(image_gray.shape[0:2], dtype=np.uint8)
    image_gray[image_gray > thresh] = 255 
    bimask = image_gray
    bimask = Image.fromarray(bimask)
    bimask = np.asarray(bimask.convert('1')).astype(np.uint8)
    annotations.append(bimask)
    return annotations
    

def create_image_info(image_id, file_name, image_size, 
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):
    dict_images = {
        "id": image_id,
        "file_name": file_name,
        "width": image_size[1],
        "height": image_size[0],
        "date_captured": date_captured,
        "license": license_id,
        "coco_url": coco_url,
        "flickr_url": flickr_url,
    }
    return dict_images


def create_annotation_info(annotation_id, image_id, binary_mask,
                           class_id, is_crowd, image_size, 
                           tolerance=2, bounding_box=None):
    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

    area = mask.area(binary_mask_encoded)
    if area < 1:
        return None

    if bounding_box is None:
        bounding_box = mask.toBbox(binary_mask_encoded)

    if is_crowd:
        segmentation = binary_mask_to_rle(binary_mask)
    else :
        segmentation = binary_mask_to_polygon(binary_mask, tolerance)
        if not segmentation:
            return None

    dict_annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": class_id,
        "iscrowd": is_crowd,
        "area": area.tolist(),
        "bbox": bounding_box.tolist(),
        "segmentation": segmentation,
        "width": image_size[1],
        "height": image_size[0]
    }
    return dict_annotation_info


def build_coco_format(images, annotations, dataset_path, subset_dir,
                      info=INFO, licenses=LICENSES, categories=CATEGORIES):
    """ Build dataset annotations in COCO Dataset format """        
    json_data = {
        'info': INFO,
        'licenses': LICENSES,
        'categories': CATEGORIES,
        'images': images,
        'annotations': annotations    
    }
    dir_annotations = os.path.join(dataset_path,'annotations')
    if not os.path.exists(dir_annotations): os.makedirs(dir_annotations)
    path_json = os.path.join(dir_annotations, 'instances_{}.json'.format(subset_dir))
    with open(path_json, 'w') as f:
       json.dump(json_data, f)
    print("JSON created...OK in:  %s", path_json)
    return json_data