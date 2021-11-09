# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 15:27:20 2021

@author: Jonat
"""
                
        

#Script for initiating training
if __name__=='__main__':
    import TrainModel
    from src.data.dataset import ColaBeerDataset

    ######## Dataset ########################
    #Specify location of training data and load the data using dataloader
    data_location = "../../data/train"
    dataset=ColaBeerDataset(data_location)
    
    m=TrainModel.trainandeval()
    
    m.train(dataset)
