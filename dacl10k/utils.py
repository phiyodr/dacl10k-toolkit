import json
from glob import glob as gglob
from os.path import join as joiner
from os import makedirs
from skimage.transform import resize
from pathlib import Path 
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np


###############################
# misc
###############################

TARGET_LIST = ['Crack', 'ACrack', 'Wetspot', 'Efflorescence', 'Rust', 'Rockpocket', 'Hollowareas', 'Cavity',
               'Spalling', 'Graffiti', 'Weathering', 'Restformwork', 'ExposedRebars', 
               'Bearing', 'EJoint', 'Drainage', 'PEquipment', 'JTape', 'WConccor']

def open_json(file):
    with open(file) as f:
        d = json.load(f)
    return d

def save_dict(dct, file):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(dct, f, indent=2)



###############################
# labelme2mask
###############################

def labelme2mask(data):
    """Takes a labelme dict and transforms the shapes to a 3D mask with shape (height,width,channels), 
    where channels are the target classes from `TARGET_LIST` (in that order).
    
    data: Filename to annotation file (str) or loaded annotation data (dict).
    
    mask: numpy.array with shape (heigt, width, channels)
    """
    TARGET_LIST = ['Crack', 'ACrack', 'Wetspot', 'Efflorescence', 'Rust', 'Rockpocket', 'Hollowareas', 'Cavity',
               'Spalling', 'Graffiti', 'Weathering', 'Restformwork', 'ExposedRebars', 
               'Bearing', 'EJoint', 'Drainage', 'PEquipment', 'JTape', 'WConccor']

    if isinstance(data, str):
        with open(data, 'r') as f:
            data = json.load(f)
    assert type(data) == dict
        
    target_dict = dict(zip(TARGET_LIST, range(len(TARGET_LIST))))
    target_mask = np.zeros((data["imageHeight"], data["imageWidth"], len(TARGET_LIST)))
    for index, shape in enumerate(data["shapes"]):
        target_img = Image.new('L', (data["imageWidth"], data["imageHeight"]), 0)
        if shape["label"] in TARGET_LIST:
            target_index = target_dict[shape["label"]]
            polygon = [(x,y) for x, y in shape["points"]] # list to tuple
            ImageDraw.Draw(target_img).polygon(polygon, outline=1, fill=1)
            target_mask[:,:,target_index] += np.array(target_img)               
    return target_mask.astype(bool).astype(np.uint8)  


###############################
# resize images
###############################


def resize_images(source_folder, target_folder, size=(512,512)):
    """Resize all images in `source_folder` to size `size` and store it in `target_folder`"""
    image_files = sorted(gglob(joiner(source_folder, "*.jpg")))
    makedirs(target_folder, exist_ok = True)
    
    for image_filename in tqdm(image_files):
        base_filename = Path(image_filename).name
        target_name = joiner(target_folder, base_filename)
        with Image.open(image_filename) as img:
            img = img.resize(size)
            img = img.save(target_name)
    print(f"Resized images saved in {target_folder}")


###############################
# resize annotations
###############################

def _resize_annotation_data(data, width_factor, height_factor):
    w_index, h_index = 0, 1

    for shape_idx in range(len(data["shapes"])):
        point_len = len(data["shapes"][shape_idx]["points"])
        for point_idx in range(point_len):
            the_w = data["shapes"][shape_idx]["points"][point_idx][w_index]
            the_h = data["shapes"][shape_idx]["points"][point_idx][h_index]

            data["shapes"][shape_idx]["points"][point_idx][w_index] = the_w / width_factor
            data["shapes"][shape_idx]["points"][point_idx][h_index] = the_h / height_factor
    return data

def resize_annotations(source_folder, target_folder, size=(512,512)):
    """Resize all labelme annotations (all polygone points and imageWiedth and imageHeight) in `source_folder` to size `size` and store it in `target_folder`"""
    annotation_files = sorted(gglob(joiner(source_folder, "*.json")))
    makedirs(target_folder, exist_ok = True)

    for annotation_filename in tqdm(annotation_files):
        data = open_json(annotation_filename)
        filename = Path(annotation_filename).name
        target_filename = joiner(target_folder, filename)
        
        width_factor = data["imageWidth"] / size[0]
        height_factor = data["imageHeight"] / size[1]

        data = _resize_annotation_data(data, width_factor, height_factor)
        data["imageWidth"] = size[0]
        data["imageHeight"] = size[1]
        save_dict(data, target_filename)
