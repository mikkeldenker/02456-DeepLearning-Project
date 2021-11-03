import os
import torch
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET


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
        self.imgs  = list(sorted(os.listdir(os.path.join(self.path, "Images"))))
    def __len__(self):
        print(len(self.files))
        return len(self.files)

    def __getitem__(self, idx: int):
        annotation_path = os.path.join(self.path, "Annotations", self.files[idx])
        img_path = os.path.join(self.path, "Images", self.imgs[idx])
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        frame_file_name = root.findall("filename")[0].text
        img = Image.open(img_path).convert("RGB") # TODO: actually open image

        targets = {
                "boxes": [],
                "labels": [],
                "area": [],
                }

        image_objects = root.findall("object")

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

        #targets = {k: torch.tensor(v) for (k, v) in targets.items()} # for the test in the bottom this needs to be outcommented
                                                                      # As you cannot plot tensor bb boxes.                      
        return img, targets


if __name__ == '__main__':
    from pprint import pprint
    i=1705
    dataset = ColaBeerDataset("../../data/")
    image_anno=dataset[i][0]
    img_bbox = ImageDraw.Draw(image_anno)
    for bbox in dataset[i][1]["boxes"]:
        
        img_bbox.rectangle(bbox, outline="green") 
    
image_anno.show()

