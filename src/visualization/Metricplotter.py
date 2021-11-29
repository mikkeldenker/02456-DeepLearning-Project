# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 10:13:21 2021

@author: Jonat
"""
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os


def metricplotting(path):
    with open(path+'/val_loss.txt', 'rb') as f:
        val_loss= pickle.load(f)
    with open(path+'/train_loss.txt', 'rb') as f:
        train_loss= pickle.load(f)
    with open(path+'/val_acc.txt', 'rb') as f:
        val_acc= pickle.load(f)
    
    epoch=np.arange(1,len(val_loss)+1)
    
    
    f, ax = plt.subplots(ncols=2,figsize=(12,12))
    ax[0].plot(epoch,val_loss,label="Validation")
    ax[0].plot(epoch,train_loss,label="Training")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].grid()
    ax[0].set_title("Validation and training loss")
    ax[0].legend(shadow=True, fancybox=True)
    
    ax[1].plot(epoch,val_acc,label="Validation")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].grid()
    ax[1].set_title("Validation accuracy")
    ax[1].legend(shadow=True, fancybox=True)
    
    
    f.tight_layout(pad=3.0)
    plt.suptitle(os.path.basename(os.path.normpath(path)),fontsize=20)
    
    plt.show()
    
metricplotting("../../reports/mobile_v3")