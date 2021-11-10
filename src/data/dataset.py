import os
import torch
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import numpy as np

LABELS = {
        "beer": 1,
        "coke": 2,
        "cola": 2,
        }


class ColaBeerDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transforms = None):
        self.path = folder_path
        self.transforms = transforms
        self.files = list(sorted(os.listdir(os.path.join(self.path, "Annotations"))))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        annotation_path = os.path.join(self.path, "Annotations", self.files[idx])
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        frame_file_name = root.findall("filename")[0].text.lower()
        img_path = os.path.join(self.path, "Images", frame_file_name)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        img = torch.tensor(img)/255
        img = img.permute(2,0,1)
        
        targets = {
                "boxes": [],
                "labels": [],
                "area": [],
                'image_id': idx,
                'iscrowd': [],
                }

        image_objects = root.findall("object")
        if len(image_objects)==0:
            targets['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            targets['labels'].append(0)
            targets['area'].append(0)
            targets['iscrowd'].append(False)

        for obj in image_objects:
            name = obj.findall("name")[0].text
            bbox = obj.findall("bndbox")[0]
            xmin = float(bbox.findall('xmin')[0].text)
            ymin = float(bbox.findall('ymin')[0].text)
            xmax = float(bbox.findall('xmax')[0].text)
            ymax = float(bbox.findall('ymax')[0].text)

            area = (xmax - xmin) * (ymax - ymin) # do we use this?

            label = LABELS[name]

            targets['boxes'].append([xmin, ymin, xmax, ymax])
            targets['labels'].append(label)
            targets['area'].append(area)
            targets['iscrowd'].append(False)

        targets = {k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v for (k, v) in targets.items()} # for the test in the bottom this needs to be outcommented
        return img, targets



if __name__ == '__main__':
    from pprint import pprint
    i=680
    dataset = ColaBeerDataset("../../data/train")
    image_anno=dataset[i][0]
    img_bbox = ImageDraw.Draw(image_anno)
    for bbox in dataset[i][1]["boxes"]:

        
         img_bbox.rectangle(bbox, outline="green") 
    

    image_anno.show()
