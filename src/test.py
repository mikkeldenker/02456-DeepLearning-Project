import torchvision
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import nms
import cv2
import numpy as np
from data.dataset import ColaBeerDataset
import time

torch.backends.quantized.engine = 'qnnpack'

if __name__ == "__main__":
    # backbone = torchvision.models.mobilenet_v3_small(pretrained=True).features
    backbone = torchvision.models.mobilenet_v3_large(pretrained=True).features
    # backbone.out_channels = 576
    backbone.out_channels = 960
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
                       rpn_score_thresh=0.7,
                       )
    model.load_state_dict(torch.load("../model_v3_large.pth"))
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear, torch.nn.BatchNorm2d})

    model.eval()

    dataset = ColaBeerDataset("../data/test")
    # dataset = torch.utils.data.Subset(dataset, list(range(680, 1200)))

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

    with torch.no_grad():
        for frame, _ in dataset:
            t1_start = time.time()
                
            
            # Need to transpose to rescale
            frame = np.array(frame).transpose(1,2,0)      
            
            # get dimensions to resize the image 
            cv2.imshow('frame', frame)
            # transpose back
            frame = frame.transpose(2,0,1)
                        

            tensor = torch.tensor(frame)

            pred = model([tensor])[0]
            frame = frame.transpose(1, 2, 0)

            # NMS on bounding boxes
            boxes = pred.get('boxes', [])
            scores = pred.get('scores', [])
            labels = pred.get('labels', [])
            iou_thresh = 0.05
            nms_tensor = nms(boxes, scores, iou_thresh)

            boxes = [boxes[idx] for idx in nms_tensor]
            labels = [labels[idx] for idx in nms_tensor]
            scores = [scores[idx] for idx in nms_tensor]

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
            frame = frame[:, :, ::-1]
            t1_end = time.time()
            frame = cv2.putText(np.array(frame), 'Current render time: {:.2f}'.format(t1_end-t1_start), org, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            print('Elapsed time for current rendering: {:.2f}'.format(t1_end-t1_start))

            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
