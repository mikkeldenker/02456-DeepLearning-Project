# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 13:23:00 2021

@author: Jonat
"""
import torchvision
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import src.models.utils as utils
from src.models.engine import train_one_epoch, evaluate

class trainandeval(object):

    def train(self, dataset, num_epochs=1):
        torch.manual_seed(100)
        num_classes=3
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f"Training on {device}")

        # split the dataset in train and test set
        indices = torch.randperm(len(dataset)).tolist()
        dataset_train = torch.utils.data.Subset(dataset, indices[:-int(len(dataset)*(4/5))+1])
        dataset_val = torch.utils.data.Subset(dataset, indices[-int(len(dataset)*(1/5))+1:])
        
        
        # Specify data loaders
        data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
        
        data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
        
        ##########################################
        
        
        ######### load a model pre-trained on COCO ############
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        backbone.out_channels = 1280
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                        output_size=7,
                                                        sampling_ratio=2)
        # put the pieces together inside a FasterRCNN model
        model = FasterRCNN(backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler)
        model.to(device)
        ###########################################################
        
        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1) 
        
        
        ########## Train the actual model ###############

        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            # evaluate(model, data_loader_val, device=device)

        return model
if __name__ == "__main__":
    trainandeval()
