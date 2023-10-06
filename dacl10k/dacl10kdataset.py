from glob import glob
from PIL import Image, ImageDraw
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import json
import random
from itertools import compress
import pandas as pd
from collections import Counter
from skimage.transform import resize
from shapely.geometry import Polygon
import pickle 
import os 
from tqdm.contrib.concurrent import process_map  #  https://stackoverflow.com/a/59905309
import psutil
from datetime import datetime

from joblib import Parallel, delayed
import contextlib
import joblib
from tqdm import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

class Dacl10kDataset(Dataset):

    TARGET_LIST = ['Crack', 'ACrack', 'Wetspot', 'Efflorescence', 'Rust', 'Rockpocket', 'Hollowareas', 'Cavity',
               'Spalling', 'Graffiti', 'Weathering', 'Restformwork', 'ExposedRebars', 
               'Bearing', 'EJoint', 'Drainage', 'PEquipment', 'JTape', 'WConccor']

    def __init__(self, split, data_path, resize_mask=(512,512), resize_img=(512,512), normalize_img=True):
        """Dataset for dacl10k dataset.
        
        Args:
            split: Only works for demo v0. "train", "valid", "test". Currenlty it is best to use "all" and then update self.df accordingly. 
            data_path: Path to dataset root
            resize_mask: Apply mask resize. You need to define the same size as with image. Default (512, 512). 
            resize_img: Apply image resize. You need to define the same size as with mask. Default (512, 512). 
        """
        self.split = split

        # paths
        self.data_path = data_path
        self.annotation_path = f"{data_path}/annotations/{self.split}"
        self.annotation_files = sorted(glob(self.annotation_path + "/*.json"))
        if len(self.annotation_files) > 0:
            print(f"Found {len(self.annotation_files)} annotation_files in folder {self.annotation_path}") 
        else:
            raise Exception(f"No annotation_files in folder {self.annotation_path}")

        self.image_path = f"{self.data_path}/images/{split}"
        self.image_files = sorted(glob(self.image_path + "/*.jpg"))
        if len(self.image_files) > 0:
            print(f"Found {len(self.image_files)} image_files in folder {self.image_path}") 
        else:
            raise Exception(f"No image_files in folder {self.image_path}")

        # resize and normalize
        self.resize_mask = resize_mask
        self.resize_img = resize_img
        self.normalize_img = normalize_img
        self.normalize_fct = transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        # prefeching can only be enabled running function `run_prefetching`
        self.use_prefetched_data = False

        self.df = self._create_df(self.annotation_files)
        self.n_samples = self.__len__()
        self.target_dict = dict(zip(self.TARGET_LIST, range(len(self.TARGET_LIST))))


    def run_prefetching(self, n_jobs=8):
        print("Prefetching started")
        self.use_prefetched_data = True
        assert self.df["imageName"].duplicated().any() == False, "There are duplicated images in self.df. Do not run `multilabel_oversampling.MultilabelOversampler` before `prefetch`!"

        def prefetcher(index):            
            img, mask, image_name = self._getitem(index, return_name=True)
            # resize
            if self.resize_img:
                img = resize(img, self.resize_img)  
                img = np.array(img* 255, dtype=np.uint8)  
            if self.resize_mask: 
                mask = resize(mask, self.resize_mask)
                mask = np.array(mask * 255, dtype=np.uint8)
            return {image_name: {"image": img, "mask": mask}}
 
        with tqdm_joblib(tqdm(desc="Prefetching", total=self.n_samples)) as progress_bar:
            results = Parallel(n_jobs=n_jobs)(delayed(prefetcher)(i) for i in range(self.n_samples))

        result_dict = {}
        for data in results: 
           result_dict.update(data)        
        self.prefetched_data = result_dict
        print("Prefeching done. Data stored in `self.prefetched_data`.")

    def save_prefetched_data(self, base_path, filename=None):
        print("Start saving prefeched data.")
        if not filename:
            filename = f"{self.split}.pkl"
        os.makedirs(base_path, exist_ok=True)
        save_path = os.path.join(base_path, filename)

        with open(save_path, 'wb') as handle:
            pickle.dump(self.prefetched_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Prefetched data saved in {save_path}.")

   
    def load_prefetched_data(self, base_path, filename):
        self.use_prefetched_data = True
        load_path = os.path.join(base_path, filename)
        print(f"Load prefeched data from {load_path}. Prefeching is now enabled.")

        with open(load_path, 'rb') as handle:
            self.prefetched_data = pickle.load(handle)
        print(f"Prefetched data loaded from {load_path}.")


    def __getitem__(self, index):
        # Load data sample 
        if self.use_prefetched_data:
            image_name = self.df.iloc[index]["imageName"]
            data = self.prefetched_data[image_name]
            img, mask = data["image"], data["mask"]
        else:
            img, mask = self._getitem(index)
          
        # resize
        if self.resize_img:
            img = resize(img, self.resize_img)    
        if self.resize_mask: 
            mask = resize(mask, self.resize_mask)

        # img transforms 
        img = transforms.ToTensor()(img) 
        img = img.to(dtype=torch.float32)

        # mask transforms
        mask = transforms.ToTensor()(mask) 
        mask = mask.to(dtype=torch.float32)
        mask = (mask > 0).float()

        if self.normalize_img:
            img = self.normalize_fct(img)
        return img, mask

    def _getitem(self, index, return_name=False): 
        """
        Get item with `index` from internal DataFrame (`self.df`).
        Image is loaded using `PIL.Image`.

        """
        image_name = self.df.iloc[index]["imageName"] # Use .iloc to get ith index row (.loc is operating on index id)
        image_filename = os.path.join(self.image_path, image_name)
        annot_filename = self.get_full_annotation_filename(image_name, self.annotation_path)

        # Image loading and transform
        img = Image.open(image_filename)
        img = np.array(img, dtype=np.uint8)

        # Get annotation details
        data = self._get_data(annot_filename)
        target_mask = self._make_mask_per_class(data)
        target_mask = (target_mask > 0).astype(np.uint8)  
    
        if return_name:
            return img, target_mask, image_name
        else:      
            return img, target_mask    

    def __len__(self):
        return self.df.shape[0]

    def _make_mask_per_class(self, data):
        """Creates a 3d mask with as many channels as classes in TARGET_LIST, i.e. with shape (h,w,c)."""
        target_mask = np.zeros((data["imageHeight"], data["imageWidth"], len(self.TARGET_LIST)))
        for index, shape in enumerate(data["shapes"]):
            target_img = Image.new('L', (data["imageWidth"], data["imageHeight"]), 0)
            if shape["label"] in self.TARGET_LIST:
                target_index = self.target_dict[shape["label"]]
                polygon = [(x,y) for x, y in shape["points"]] # list to tuple
                ImageDraw.Draw(target_img).polygon(polygon, outline=1, fill=1)
                target_mask[:,:,target_index] += np.array(target_img)               
        return target_mask.astype(bool).astype(np.uint8)        

    @staticmethod
    def _get_data(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data

    def _create_df(self, annotation_paths):
        """Create DataFrame for easier analysis of dacl5k data."""

        # Empty dataFrame
        df = pd.DataFrame()

        # Empty counter, e.g. Counter({'Rust': 7, 'Spalling': 3, 'Crack': 0, 'Efflorescence': 0, 'ExposedRebars': 0})
        empty_counter = Counter(dict(zip(self.TARGET_LIST, [0] * len(self.TARGET_LIST)))) 

        for filename in tqdm(annotation_paths, desc="Create internal df"):
            # Get data and add new "class" key
            data = self._get_data(filename)
            class_list = [x["label"] for x in data["shapes"]]
            # Count damages
            counts = Counter(class_list)
            counts.update(empty_counter)

            # Count damages raw
            label_list = [x["label"] for x in data["shapes"]]
            counts_raw = Counter(label_list)

            # Remove shapes update counts
            data.pop("shapes")
            data.update(counts)

            # Append current data to df
            data_df = pd.DataFrame([data])
            df = pd.concat([df, data_df], ignore_index=True)
        print("DataFrame is stored in `self.df`.")
        return df

    @staticmethod
    def get_full_annotation_filename(image_name, annotation_path):
        base_name = image_name.split(".")[0]
        filename = base_name + ".json"
        full_annotation_filename = os.path.join(annotation_path, filename)
        return full_annotation_filename
        
    def __repr__(self):
        return f"Dacl5kDataset(split={str(self.split or 'None')}, data_path={self.data_path}, resize_mask={self.resize_mask}, resize_img={self.resize_img}, normalize_img={self.normalize_img})"
        

if __name__ == "__main__":   
    now = datetime.now()
    print(now)     
    PATH_TO_DATA = "/home/philipp/Documents/project_dacl5kSegSeg/dacl5k/data/dacl5k/v1.0"

    split = "validation"
    resize_mask = (512, 512) 
    resize_img = (512, 512)

    for split in ["validation"]:
        print("="*80)
        print(split)
        dataset = Dacl10kDataset(split, PATH_TO_DATA, resize_mask=resize_mask, resize_img=resize_img, normalize_img=True)
        #dataset = Dacl10kDataset(split, PATH_TO_DATA, resize_mask=resize_mask, resize_img=resize_img, normalize_img=True)

        dataset[0]
        import pdb; pdb.set_trace()
        #dataset.run_prefetching(n_jobs=10)
        #dataset.save_prefetched_data(base_path="/home/philipp/Desktop", filename="validation.pkl")
        print("="*80)
    now = datetime.now()
    print(now)    
    print("Done")       

