import torchvision
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import cv2
import numpy as np
from src.data.dataset import ColaBeerDataset


if __name__ == "__main__":
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)
# put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                       num_classes=3,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    model.load_state_dict(torch.load("../model.pth"))
    model.eval()

    dataset=ColaBeerDataset("../data/train")

    with torch.no_grad():
        for frame, _ in dataset:
            frame = np.array(frame)
            tensor = torch.tensor(frame)
            tensor = tensor / 255
            pred = model([tensor])[0]
            frame = frame.transpose(1, 2, 0)

            for i, box in enumerate(pred['boxes']):
                xmin, ymin, xmax, ymax = box
                xmin = int(xmin)
                xmax = int(xmax)
                ymin = int(ymin)
                ymax = int(ymax)

                print(xmin, ymin, xmax, ymax)

                label = int(pred['labels'][i])
                if label == 1:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)

                frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3)
          
            # Display the resulting frame
            frame = frame[:, :, ::-1]
            cv2.imshow('frame', frame)
            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
