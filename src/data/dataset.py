import os
import torch
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import numpy as np
import torchvision.transforms as T

LABELS = {
        "beer": 1,
        "coke": 2,
        "cola": 2,
        }


class ColaBeerDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, apply_transforms):
        self.path = folder_path
        self.files = list(sorted(os.listdir(os.path.join(self.path, "Annotations"))))
        self.apply_transforms = apply_transforms
        self._color_transform = T.ColorJitter(1, 0.1, 0.1, 0.05)


    def __len__(self):
        return len(self.files)


    def _apply_transforms(self, image, targets):
        if not self.apply_transforms:
            return image, targets

        image = self._color_transform(image)
        return image, targets


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
        img, targets = self._apply_transforms(img, targets)
        return img, targets


class Combined(torch.utils.data.Dataset):
    def __init__(self, paths, apply_augmentations=True):
        self._datasets = [ColaBeerDataset(p, False) for p in paths]
        if apply_augmentations:
             self._datasets = self._datasets + [ColaBeerDataset(p, True) for p in paths]

    def __len__(self):
        return sum([len(subset) for subset in self._datasets])

    def __getitem__(self, idx):
        current_idx = 0

        while len(self._datasets[current_idx]) <= idx:
            idx -= len(self._datasets[current_idx])
            current_idx += 1

        return self._datasets[current_idx][idx]

if __name__ == '__main__':
    import math
    combined = Combined(['../../data/train', '../../data/test'], True)
    train_percent = 0.8
    train_dataset, val_dataset = torch.utils.data.random_split(combined, [math.ceil(len(combined)*train_percent), math.floor((1-train_percent)*len(combined))])
    print(train_dataset, len(train_dataset))

    import cv2
    for frame, _ in train_dataset:
        frame = np.array(frame)
        frame = frame.transpose(1, 2, 0)
        frame = frame[:, :, ::-1]
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

