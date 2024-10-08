import os
import gcsfs
import tarfile
import torch
import numpy as np
from PIL import Image
from zipfile import ZipFile
from torch.utils.data import Dataset



class Clothing1M(Dataset):
    r"""https://github.com/LiJunnan1992/MLNT"""

    def __init__(self, root, mode, transform=None, percent_clean=None,
                 gcp:bool=False, local:bool=False, val_size=None):
        self.root = root
        self.anno_dir = os.path.join(self.root, "annotations")
        self.transform = transform
        self.mode = mode
        self.percent_clean = percent_clean
        self.gcp = gcp
        self.local = local
        self.val_size = val_size
        
        #Download the dataset from gcp if required:
        if self.gcp:
            self.download_dataset()

        self.imgs = []
        self.labels = {}
        
        if self.mode == "dirty_train":
            img_list_file = "clean_train_key_list.txt"
            label_list_file = "clean_label_kv.txt"
            self.img_paths(img_list_file)
            self.gen_labels(label_list_file)
            img_list_file = "clean_val_key_list.txt"
            label_list_file = "clean_label_kv.txt"
            self.img_paths(img_list_file)
            self.gen_labels(label_list_file)
            
            self.clean_count = len(self.imgs)
            if percent_clean is not None:
                self.number_noisy = int(self.clean_count * (100-percent_clean)/percent_clean)
            img_list_file = "noisy_train_key_list.txt"
            label_list_file = "noisy_label_kv.txt"
            self.img_paths(img_list_file)
            self.gen_labels(label_list_file)
        
        if self.mode == "noisy_train":
            img_list_file = "noisy_train_key_list.txt"
            label_list_file = "noisy_label_kv.txt"
            self.img_paths(img_list_file)
            self.gen_labels(label_list_file)

        if self.mode == "train":
            img_list_file = "clean_train_key_list.txt"
            label_list_file = "clean_label_kv.txt"
            self.img_paths(img_list_file)
            self.gen_labels(label_list_file)
            self.clean_count = len(self.imgs)
            if percent_clean is not None:
                self.number_noisy = int(self.clean_count * (100-percent_clean)/percent_clean)
            img_list_file = "noisy_train_key_list.txt"
            label_list_file = "noisy_label_kv.txt"
            self.img_paths(img_list_file)
            self.gen_labels(label_list_file)
            
            if percent_clean is not None:
                state = np.random.get_state()
                np.random.seed(0)
                noisy_selection = np.random.choice(np.arange(self.clean_count, len(self.imgs)), size=self.number_noisy, replace=False)
                np.random.set_state(state)
                self.imgs = np.append(np.array(self.imgs[:self.clean_count]), np.array(self.imgs)[noisy_selection])
                self.clean_indicator = np.zeros(len(self.imgs))
                self.clean_indicator[:self.clean_count] = 1

        if self.mode == "dirty_val":
            
            assert self.val_size is not None, 'for mode=="dirty_val", val_size must not be None'
            
            all_set = np.arange(1061883)
            val_subset = np.random.choice(all_set, size=self.val_size, replace=False)
            
            #This is the 'dirty train' mode code:
            img_list_file = "clean_train_key_list.txt"
            label_list_file = "clean_label_kv.txt"
            self.img_paths(img_list_file)
            self.gen_labels(label_list_file)
            img_list_file = "clean_val_key_list.txt"
            label_list_file = "clean_label_kv.txt"
            self.img_paths(img_list_file)
            self.gen_labels(label_list_file)
            
            self.clean_count = len(self.imgs)
            if percent_clean is not None:
                self.number_noisy = int(self.clean_count * (100-percent_clean)/percent_clean)
            img_list_file = "noisy_train_key_list.txt"
            label_list_file = "noisy_label_kv.txt"
            self.img_paths(img_list_file)
            self.gen_labels(label_list_file)
            
            #We want sample only a subset of the dataset:
            self.imgs = np.array(self.imgs)
            self.imgs = self.imgs[val_subset].tolist()    
            
            #We now add the remaining validation datasamples:
            img_list_file = "clean_val_key_list.txt"
            label_list_file = "clean_label_kv.txt"
            self.img_paths(img_list_file)
            self.gen_labels(label_list_file)
                

        if self.mode == "val":
            
            img_list_file = "clean_val_key_list.txt"
            label_list_file = "clean_label_kv.txt"
            self.img_paths(img_list_file)
            self.gen_labels(label_list_file)
            
        elif self.mode == "test":
            
            img_list_file = "clean_test_key_list.txt"
            label_list_file = "clean_label_kv.txt"
            self.img_paths(img_list_file)
            self.gen_labels(label_list_file)

        self.classes = [
            "T-Shirt",
            "Shirt",
            "Knitwear",
            "Chiffon",
            "Sweater",
            "Hoodie",
            "Windbreaker",
            "Jacket",
            "Downcoat",
            "Suit",
            "Shawl",
            "Dress",
            "Vest",
            "Underwear",
        ]
        self.sequence = np.arange(len(self.imgs))
        
        #Setup targets as a list formatted set of labels
        self.targets = list()
        for img in self.imgs:
            self.targets.append(self.labels[img])
                
                
        #If local use the io loader else load the data directly into memory:
        if not self.local:
            pass
            
            
    def download_dataset(self):
        
        #Hard coded GCP locations
        fs = gcsfs.GCSFileSystem(project='robust-active-subsampling') 
        gcp_storage = "gs://robust-active-subsampling-xmanager-test-bucket"
        
        #install the annotations file:
        path = os.path.join(gcp_storage, "datasets", 
                            "clothing1M", "annotations.zip")
        
        with fs.open(path, 'rb') as f:
          with ZipFile(f, 'r') as z:
            z.extractall(self.anno_dir)
        
        print([f for f in os.listdir(self.root) if os.path.isfile(os.path.join(self.root, f))])
        
        for i in range(10):

            #Download the individual tar files from gcp:
            path = os.path.join(gcp_storage, "datasets", 
                                "clothing1M", "images",
                                f'{i}.tar')
            
            extract_path = os.path.join(self.root, 'images')
            
            #Open tar files and save locally to the anno_dir:
            with fs.open(path, 'rb') as f:
              tr = tarfile.open(fileobj=f, mode='r:')
              tr.extractall(extract_path)
             
                

    def indicate_clean(self, indices):
        if self.percent_clean is not None:
            return self.clean_indicator[indices.cpu().numpy()]
        else:
            return 0
        
    def img_paths(self, img_list_file):
        
        #Read file from local path
        with open(os.path.join(self.anno_dir, img_list_file), "r") as f:
            lines = f.read().splitlines()
                
        #Add img paths self.imgs
        for l in lines:
            self.imgs.append(l)

                
    def gen_labels(self, label_list_file):
        
        #Read file from local path:
        if not self.gcp:
            with open(os.path.join(self.anno_dir, label_list_file), "r") as f:
                lines = f.read().splitlines()
            
        #Read file from gcp
        else:
            gcp_storage = "gs://robust-active-subsampling-xmanager-test-bucket"
            fs = gcsfs.GCSFileSystem(project='robust-active-subsampling')
            with fs.open(os.path.join(gcp_storage, "datasets", 
                                      "clothing1M", "annotations",
                                      label_list_file), "r") as f:
                lines = f.read().splitlines()
              
        #Process each line in lines
        for l in lines:
            entry = l.split()
            #img_path = os.path.join(self.root, entry[0])
            self.labels[entry[0]] = int(entry[1])
            
            
    def load_image(self, image_path):
        return Image.open(os.path.join(self.root, image_path)).convert('RGB')


    def __getitem__(self, index):
        
        idx = self.sequence[index]
        img_path = self.imgs[idx]
        target = self.targets[idx]
        #For now the target is also returned as the 'group' g

        target = torch.tensor(target)
        idx = torch.tensor(idx)


        image = self.load_image(img_path)
        #image = Image.open(img_path).convert("RGB")
        img = self.transform(image)
        
        return idx.long(), img, target, target

    def __len__(self):
        return len(self.sequence)
