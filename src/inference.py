import torchvision
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import cv2
import numpy as np


if __name__ == "__main__":
    model = torch.jit.load("../traced.pth")
    model.eval()

    with torch.no_grad():
        vid = cv2.VideoCapture(0)
  
        while(True):
              
            # Capture the video frame
            # by frame
            ret, frame = vid.read()
            frame = cv2.flip(frame, 1)

            frame = np.array(frame)
            frame = frame[:, :, ::-1]
            tensor = torch.tensor(frame.copy()).permute(2, 0, 1)
            tensor = tensor / 255
            print(tensor.shape)
            pred = model([tensor])[0]
            print(pred)
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
            cv2.imshow('frame', frame)
            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

