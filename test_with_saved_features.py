import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import json
import torch.utils.data.sampler
import os
import glob
import random
import time

import configs
import backbone
import data.feature_loader as feat_loader
from data.datamgr import SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet

from io_utils import model_dict, parse_args, get_resume_file, get_best_file , get_assigned_file

from datasets import ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot

def feature_evaluation(cl_data_file, model, n_way = 5, n_support = 5, n_query = 15, adaptation = False):
    class_list = cl_data_file.keys()
    select_class = random.sample(class_list, n_way)
    z_all  = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append( [ np.squeeze( img_feat[perm_ids[i]]) for i in range(n_support+n_query) ] )     # stack each batch

    z_all = torch.from_numpy(np.array(z_all) )
   
    model.n_query = n_query
    if adaptation:
        scores  = model.set_forward_adaptation(z_all, is_feature = True)
    else:
        scores  = model.set_forward(z_all, is_feature = True)

        
    pred = scores.data.cpu().numpy().argmax(axis = 1)
    y = np.repeat(range( n_way ), n_query )
    acc = np.mean(pred == y)*100 
    return acc

if __name__ == '__main__':
    params = parse_args('test')

    acc_all = []

    iter_num = 600
    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot) 

    if params.method == 'baseline':
        model           = BaselineFinetune( model_dict[params.model], **few_shot_params )
    elif params.method == 'protonet':
        model           = ProtoNet( model_dict[params.model], **few_shot_params )
    else: 
       raise ValueError('Unknown method')


    model = model.cuda()
    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, 'miniImageNet', params.model, params.method)
    if params.train_aug:
        checkpoint_dir += '_aug'

    if not params.method in ['baseline'] :
        checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    if not params.method in ['baseline'] : 
        if params.save_iter != -1:
            modelfile   = get_assigned_file(checkpoint_dir,params.save_iter)
        else:
            modelfile   = get_best_file(checkpoint_dir)
            
        if modelfile is not None:
            tmp = torch.load(modelfile)
            model.load_state_dict(tmp['state'])


    #params.save_iter = 399
    if params.save_iter != -1:
        novel_file = os.path.join( checkpoint_dir.replace("checkpoints","features"), params.dataset + "_" + str(params.save_iter)+".hdf5") #defaut split = novel, but you can also test base or val classes
    else:
        novel_file = os.path.join( checkpoint_dir.replace("checkpoints","features"), params.dataset + ".hdf5") #defaut split = novel, but you can also test base or val classes


    cl_data_file = feat_loader.init_loader(novel_file)

    for i in range(iter_num):
        print (i)
        acc = feature_evaluation(cl_data_file, model, n_query = 15, adaptation = params.adaptation, **few_shot_params)
        print (acc)
        acc_all.append(acc)

    acc_all  = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
