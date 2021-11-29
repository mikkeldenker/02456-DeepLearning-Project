import torchvision
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import batched_nms
import cv2
import numpy as np
import time

torch.backends.quantized.engine = 'qnnpack'

if __name__ == "__main__":
    #model = torch.jit.load("../traced.pth")
    backbone = torchvision.models.mobilenet_v3_small(pretrained=True).features
    backbone.out_channels = 576
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
                   rpn_score_thresh=0.2,
                   )
    model.load_state_dict(torch.load("../models/model_v3.pth"))
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear, torch.nn.BatchNorm2d})
    # model = torch.jit.script(model)
    #model.half()
    model.eval()
    
    # Scale percentage for inference
    scale_percentage = 50
    
    # data for text
    # Window name in which image is displayed
    window_name = 'Image'
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (50, 50)
    # fontScale
    fontScale = 1
    # black color in BGR
    color = (255, 255, 255)
    # Line thickness of 2 px
    thickness = 2

    with torch.no_grad():
        vid = cv2.VideoCapture(0)
  
        while(True):
            t1_start = time.time()
            
            # Capture the video frame
            # by frame
            ret, frame = vid.read()
            frame = cv2.flip(frame, 1)
            
            # get dimensions to resize the image 
            width = int(frame.shape[1]*scale_percentage/100)
            height = int(frame.shape[0]*scale_percentage/100)
            dim = (width, height)
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            
            
            frame = np.array(frame)
            frame = frame[:, :, ::-1]
            tensor = torch.tensor(frame.copy()).permute(2, 0, 1)
            tensor = tensor/255
            pred = model([tensor])[0]
            frame = frame[:, :, ::-1]

            for i, box in enumerate(pred.get('boxes', [])):
                xmin, ymin, xmax, ymax = box
                xmin = int(xmin)
                xmax = int(xmax)
                ymin = int(ymin)
                ymax = int(ymax)

                label = int(pred['labels'][i])
                if label == 1:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 5)
                
                
            # Display the resulting frame
            t1_end = time.time()
            color = (0, 0, 0)
            cv2.putText(frame, 'Current render time: {:.2f}'.format(t1_end-t1_start), org, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            print('Elapsed time for current rendering: {:.2f}'.format(t1_end-t1_start))

            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
