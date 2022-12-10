#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.clear()
sys.path.append("/home/toor/Desktop/doc_sco/image/dataset_of_github/main/parking-space-occupancy-main")

print(sys.path)
import os
import torch
import torchvision




from utils.dataset import acpds
from utils.utils import transforms
from utils.utils import visualize as vis

train_ds, valid_ds, test_ds = acpds.create_datasets('./dataset/data')


def image_pt_to_np(image):
    """
    Convert a PyTorch image to a NumPy image (in the OpenCV format).
    """
    image = image.cpu().clone()
    image = image.permute(1, 2, 0)
    image_mean = torch.tensor([0.485, 0.456, 0.406])
    image_sd = torch.tensor([0.229, 0.224, 0.225])
    image = image * image_sd + image_mean
    image = image.clip(0, 1)
    return image




def create_annotation(fileFullPath,index,rois,labels,name):
    fileFullPath = fileFullPath+"/labels/"
    isExist = os.path.exists(fileFullPath)
    if not isExist:
       os.makedirs(fileFullPath)
    
    with open(fileFullPath+'/'+name+"{:03d}".format(index)+'.txt', 'w+') as f:
        i=0
        for carre in rois:
            lab = str(labels[i].item()) + ' '
            #print(lab, end = '')
            
            f.write(lab)
            i+=1
            
            for coor in carre:
                
                for xy in coor:
                    #if cb==4:
                    #    element = "{:.4f}".format((xy.item()))
                    #else:
                    element = "{:.4f}".format((xy.item()))+ ' '
                    #print(element, end = '')
                    f.write(element)
            
            coor = carre[0]   
            cb = 0
            for xy in coor:
                cb+=1
                if cb==2:
                    element = "{:.4f}".format((xy.item()))
                else:
                    element = "{:.4f}".format((xy.item()))+ ' '
                #print(element, end = '')
                f.write(element)
            
            #print()
            f.write("\n")


def create_img(path,image,index,name):
    path = path+"/image/"
    isExist = os.path.exists(path)
    if not isExist:
       os.makedirs(path)
    from PIL import Image
    import numpy as np
    img = image_pt_to_np(image).numpy()
    img = img * 255
    img = img.astype(np.uint8)
    
    im = Image.fromarray(img)
    name = name+"{:03d}".format(index)+'.jpeg'
    im.save(path+name)



def create_img_anno(dataset,name):
    
    fileFullPath = '/home/toor/Desktop/doc_sco/image/dataset_of_github/main/parking-space-occupancy-main/dataset_out/'+name
    taille = len(dataset)
    print("dataset")
    print(taille)
    index=0
    isExist = os.path.exists(fileFullPath)
    if not isExist:
       os.makedirs(fileFullPath)
    
    dataiter = iter(dataset)
    
    
    print("start")
    for ele in range(taille):
    #for ele in range(1):
        
        image_raw, rois, labels = dataiter.next()
        image_raw, rois_r, labels = image_raw[0], rois[0], labels[0]
        
        
        
        for i in range(34):
            img, rois = transforms.augment(image_raw, rois_r)
            
            index+=1
            image = transforms.preprocess(img, res=640)
            #vis.plot_ds_image(image, rois, labels, show=True)
            #print(len(rois))
            print("image "+str(index))
            
            create_annotation(fileFullPath,index,rois,labels,name)
            create_img(fileFullPath,image,index,name)
      
    
create_img_anno(train_ds,'train')
create_img_anno(valid_ds,'valid')
create_img_anno(test_ds,'test')
