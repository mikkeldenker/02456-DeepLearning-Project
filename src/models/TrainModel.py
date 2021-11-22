# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 13:23:00 2021

@author: Jonat
"""
import torchvision
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.tensorboard import SummaryWriter
import src.models.utils as utils
from src.models.engine import train_one_epoch, evaluate, evaluate_mik
import pickle
class trainandeval(object):

    def train(self, dataset, num_epochs=1,model="mobilev2"):
        torch.manual_seed(100)
        num_classes=3
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f"Training on {device}")
        
        writer = SummaryWriter(
            log_dir="../../reports"
        )
        #
        # split the dataset in train and test set
        indices = torch.randperm(len(dataset)).tolist()
        dataset_train = torch.utils.data.Subset(dataset, indices[:int(len(dataset)*(4/5))+1])
        dataset_val = torch.utils.data.Subset(dataset, indices[int(len(dataset)*(/45))+1:])
        
        
        # Specify data loaders
        data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
        
        data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
        
        ##########################################
        
        
        ######### load a model pre-trained on COCO ############
        assert model in ["mobilev2", "resnet50"] , 'select either "resnet50" or "mobilev2"' 
        print(f'Training using {model}')
        if model=="mobilev2":
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
        elif model=="resnet50":
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            num_classes = 3 # cola and beer + background
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        elif model == 'mobilev3':
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
                       )
        model.to(device)
        ###########################################################
        
        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=1) 
        
        
        ########## Train the actual model ###############
        train_loss={}
        val_loss={}
        val_acc={}
        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            metric_logger=train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
            TL = metric_logger.loss.total #total loss
            count = metric_logger.loss.count
            SL = float(TL / count) #single loss or average loss
            train_loss.append(SL)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset validation acc
            evaluation_result=evaluate(model, data_loader_val, device=device)          
            test_accuracy = evaluation_result.coco_eval.get("bbox").stats[0]
            val_acc.append(test_accuracy)
            #validation loss
            test_metric=evaluate_mik(model,data_loader_val,epoch,device=device,print_freq=1)
            VL = test_metric.loss.total #total loss
            count = test_metric.loss.count
            VSL = float(VL / count) #single loss or average loss
            val_loss.append(VSL)
            print(val_loss)
        with open("val_loss.txt", "wb") as fp:   #Pickling
          pickle.dump(val_loss, fp)
        with open("train_loss.txt", "wb") as fp:   #Pickling
          pickle.dump(train_loss, fp)
        with open("val_acc.txt", "wb") as fp:   #Pickling
          pickle.dump(val_acc, fp)
            
            
            
        return model
if __name__ == "__main__":
    trainandeval()
