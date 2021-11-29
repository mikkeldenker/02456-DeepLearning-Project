import torchvision
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
import cv2
import numpy as np
from data.dataset import ColaBeerDataset, Combined
from models.centroid_tracking import CentroidTracker
import time
import math
import os

torch.backends.quantized.engine = 'qnnpack'

THRESH = 0.3

if __name__ == "__main__":
    backbone = torchvision.models.mobilenet_v3_small(pretrained=True).features
    backbone.out_channels = 576
    # backbone.out_channels = 960
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)
# put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                       num_classes=3,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler,
                       min_size=220,
                       max_size=220,
                       rpn_score_thresh=0.3,
                       )
    model.load_state_dict(torch.load("../models/model_v3_small.pth"))

    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # num_classes = 3 # cola and beer + background
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # model.load_state_dict(torch.load("../models/model_resnet50_10epoch.pth"))

    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear, torch.nn.BatchNorm2d})

    model.eval()

    torch.manual_seed(100)
    # dataset = Combined(['../data/train', '../data/test'])
    # train_percent = 0.8
    # _, dataset = torch.utils.data.random_split(dataset, [math.ceil(len(dataset)*train_percent), math.floor((1-train_percent)*len(dataset))])

    dataset = ColaBeerDataset('../data/test')
    # dataset = torch.utils.data.Subset(dataset, list(range(0, 500)))

    # data for text
    # Window name in which image is displayed
    window_name = 'Image'
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (20, 20)
    # fontScale
    fontScale = 0.5
    # blue color in BGR
    color = (255, 0, 0)
    # Line thicknes
    thickness = 2
    out = None
    if os.path.exists('output.avi'):
        os.remove('output.avi')
    tracker = CentroidTracker(24, 200)
    with torch.no_grad():
        for frame, _ in dataset:
            t1_start = time.time()
                
            if out is None:
                fourcc = cv2.VideoWriter_fourcc(*'XVID') # Be sure to use the lower case
                out = cv2.VideoWriter('output.avi', fourcc, 20, (frame.shape[2], frame.shape[1]))
            
            tensor = torch.tensor(frame)

            pred = model([tensor])[0]
            frame = np.array(frame)
            frame = frame.transpose(1, 2, 0)

            # NMS on bounding boxes
            boxes = pred.get('boxes', [])
            scores = pred.get('scores', [])
            labels = pred.get('labels', [])
            iou_thresh = 0.3
            nms_tensor = nms(boxes, scores, iou_thresh)

            scores = [scores[idx] for idx in nms_tensor]
            boxes = [boxes[idx] for i, idx in enumerate(nms_tensor) if scores[i] > THRESH]
            labels = [labels[idx] for i, idx in enumerate(nms_tensor) if scores[i] > THRESH]
            tracker.update(boxes)

            frame = (frame*255).astype(np.uint8)

            for obj_id, centroid in tracker.objects():
                print(obj_id, centroid)
                frame = cv2.circle(frame, centroid, radius=0, color=(255,127,80), thickness=10)
                frame = cv2.putText(frame, str(obj_id), centroid, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,127,80), 2)
            for (i, box) in enumerate(boxes):
                #box = pred['boxes'][i]
                xmin, ymin, xmax, ymax = box
                xmin = int(xmin)
                xmax = int(xmax)
                ymin = int(ymin)
                ymax = int(ymax)

                label = int(labels[i])
                if label == 1:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)

                frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3)
          
            
            # Display the resulting frame
            color = (255, 127, 80)
            t1_end = time.time()
            frame = cv2.putText(np.array(frame), 'Current render time: {:.2f}'.format(t1_end-t1_start), org, font, fontScale, color, thickness, cv2.LINE_AA)
            frame = frame[:, :, ::-1]
            out.write(frame)
            cv2.imshow('frame', frame)
            print('Elapsed time for current rendering: {:.2f}'.format(t1_end-t1_start))

            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
out.release()
