from torch.utils.data import Dataset
from pathlib import Path
from .labels import labels
import pandas as pd

class_info = [label.name for label in labels if label.ignoreInEval is False]
color_info = [label.color for label in labels if label.ignoreInEval is False]

color_info += [[0, 0, 0]]

map_to_id = {}
i = 0
for label in labels:
    if label.ignoreInEval is False:
        map_to_id[label.id] = i
        i += 1     

id_to_map = {id: i for i, id in map_to_id.items()}   

class DeepGlobeLLC(Dataset):
    class_info = class_info
    color_info = color_info
    num_classes = len(class_info)
    
    #stavio sam srednju vrijednost i std cijelog dataseta
    
    # mean= [104.09488634, 96.66853759 , 71.80576166] / 255
    # std = [37.4961336 , 29.24727474, 26.74629693] / 255
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    map_to_id = map_to_id
    id_to_map = id_to_map

    def __init__(self, root: Path, transforms: lambda x: x, subset='train',epoch=None):
        self.root = root
        self.labels_dir = self.root / "labels" / "ids" / subset
        self.images_dir = self.root / subset
        self.subset = subset
        self.has_labels = True
        self.transforms = transforms
        self.epoch = epoch
        
        df = pd.read_csv(root / f'{subset}.csv')
        
        self.images = [Path(p) for p in df['sat_image_path']]
        
        if self.has_labels:
            self.labels = [Path(p) for p in df['mask_path']]
        
        print(f'Num images: {len(self)}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        ret_dict = {
            'image': self.images[item],
            'name': self.images[item].stem.split('_')[0],
            'subset': self.subset
        }
        if self.has_labels:
            ret_dict["labels"] = self.labels[item]
        if self.epoch is not None:
            ret_dict["epoch"] = int(self.epoch.value)

        return self.transforms(ret_dict)
