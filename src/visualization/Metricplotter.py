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
    
    
    f, ax = plt.subplots(ncols=2,figsize=(16,16), constrained_layout=True)
    ax[0].plot(epoch,val_loss,label="Validation")
    ax[0].plot(epoch,train_loss,label="Training")
    ax[0].set_xlabel("epoch",fontsize=24)
    ax[0].set_ylabel("Loss",fontsize=24)
    ax[0].grid()
    ax[0].set_title("Validation and training loss",fontsize=24)
    ax[0].legend(shadow=True, fancybox=True,prop={'size':18})
    
    
    ax[1].plot(epoch,val_acc,label="Validation")
    ax[1].set_xlabel("epoch",fontsize=24)
    ax[1].set_ylabel("Accuracy",fontsize=24)
    ax[1].grid()
    ax[1].set_title("Validation accuracy",fontsize=24)
    ax[1].legend(shadow=True, fancybox=True,prop={'size':18})
    plt.rc('xtick', labelsize=24) 
    plt.rc('ytick', labelsize=24) 
    
    f.tight_layout(pad=6.0)
    modelname=os.path.basename(os.path.normpath(path))
    #plt.suptitle(modelname,fontsize=32,y=1.05)
    f.suptitle("\n".join([modelname]),fontsize=32, y=0.98)
    
    plt.show()
    plt.savefig("../../reports/figures/"+modelname+".pdf")
    
metricplotting("../../reports/mobile_v3")