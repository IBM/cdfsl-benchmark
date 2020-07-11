import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
from itertools import combinations

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager

from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file 

from utils import *
import backbone

from datasets import ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot,  Chest_few_shot

class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        
        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x

def train_loss_cross_validation(embeddings, y_a_i, support_size, n_support, n_way, total_epoch):

    embeddings = Variable(embeddings).cuda()
    all_losses = []

    for r in range(n_support):

        train_embeddings = []
        val_embeddings = []
        train_y = []
        val_y = []

        for idx in range(embeddings.size()[0]):
            if (idx - r) % n_support == 0:
                val_embeddings.append(embeddings[idx, :].view(1, embeddings[idx, :].size()[0]))
                val_y.append(y_a_i[idx])
            else:
                train_embeddings.append(embeddings[idx, :].view(1, embeddings[idx, :].size()[0]))
                train_y.append(y_a_i[idx])

        train_y = np.asarray(train_y)
        val_y = np.asarray(val_y)

        val_embeddings = torch.cat(val_embeddings, 0)
        train_embeddings = torch.cat(train_embeddings, 0)

        loss_fn = nn.CrossEntropyLoss().cuda()
        net = Classifier(train_embeddings.size()[1], n_way).cuda()
        classifier_opt = torch.optim.SGD(net.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
       
        train_y = Variable(torch.from_numpy(train_y)).cuda() 

        train_size = support_size - n_support
        batch_size = 4
        for epoch in range(total_epoch):
            rand_id = np.random.permutation(train_size)

            for j in range(0, train_size, batch_size):
                classifier_opt.zero_grad()

                #####################################
                selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, train_size)]).cuda()
                z_batch = train_embeddings[selected_id]

                y_batch = train_y[selected_id] 
                #####################################
                outputs = net(z_batch)            
                #####################################
        
                loss = loss_fn(outputs, y_batch)
                loss.backward()
                classifier_opt.step()

        val_y = Variable(torch.from_numpy(val_y)).cuda() # (25,)

        outputs = net(val_embeddings)   
        loss = loss_fn(outputs, val_y)
        all_losses.append(loss)

    return sum(all_losses) / (len(all_losses) + 0.0)

def train_loss_half_validation(embeddings, y_a_i, support_size, n_support, n_way, total_epoch):
    embeddings = embeddings.cpu().numpy()

    train_embeddings = []
    val_embeddings = []
    train_y = []
    val_y = []

    for idx in range(support_size):
        if (idx % 10) % 2 == 0:
            val_embeddings.append(embeddings[idx, :].reshape(1, embeddings[idx, :].shape[0]))
            val_y.append(y_a_i[idx])
        else:
            train_embeddings.append(embeddings[idx, :].reshape(1, embeddings[idx, :].shape[0]))
            train_y.append(y_a_i[idx])

    train_y = np.asarray(train_y)
    val_y = np.asarray(val_y)

    val_embeddings = torch.from_numpy(np.concatenate( val_embeddings, axis=0 ))
    train_embeddings = torch.from_numpy(np.concatenate( train_embeddings, axis=0 ))

    loss_fn = nn.CrossEntropyLoss().cuda()
    net = Classifier(train_embeddings.size()[1], n_way).cuda()

    classifier_opt = torch.optim.SGD(net.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
   
    train_y = Variable(torch.from_numpy(train_y)).cuda()
    train_embeddings = Variable(train_embeddings).cuda()

    train_size = support_size / 2
    batch_size = 4
    for epoch in range(total_epoch):
        rand_id = np.random.permutation(train_size)

        for j in range(0, train_size, batch_size):
            classifier_opt.zero_grad()

            #####################################
            selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, train_size)]).cuda()
            z_batch = train_embeddings[selected_id]

            y_batch = train_y[selected_id] 
            #####################################
            outputs = net(z_batch)            
            #####################################
    
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            classifier_opt.step()

    val_embeddings = Variable(val_embeddings).cuda()
    val_y = Variable(torch.from_numpy(val_y)).cuda() 

    outputs = net(val_embeddings)   
    loss = loss_fn(outputs, val_y)

    return loss

def combine_model(model_embeddings, y_a_i, support_size, n_support, n_way, cross_validation_epoch, with_replacement=True):

    embeddings_idx_model = []
    embeddings_all = None

    min_loss = float("inf")

    for num in range(len(model_embeddings)):
        embedding_candidate = None
        idx_candidate = -1

        for idx, embedding in enumerate(model_embeddings):

            if embeddings_all is None:

                if n_support == 20 or 50:
                    running_loss = train_loss_half_validation(embedding, y_a_i, support_size, n_support, n_way, cross_validation_epoch)
                else:
                    running_loss = train_loss_cross_validation(embeddings, y_a_i, support_size, n_support, n_way, cross_validation_epoch)

            else:
                tmp_embedding = torch.cat((embeddings_all, embedding), 1)
                
                if n_support == 20 or 50:
                    running_loss = train_loss_half_validation(tmp_embedding, y_a_i, support_size, n_support, n_way, cross_validation_epoch)

                else:
                    running_loss = train_loss_cross_validation(embeddings, y_a_i, support_size, n_support,n_way,  cross_validation_epoch)


            if running_loss < min_loss:
                embedding_candidate = embedding
                idx_candidate = idx
                min_loss = running_loss

        if with_replacement:
            if idx_candidate != -1: 
                embeddings_idx_model.append(idx_candidate)
                if embeddings_all is None:
                    embeddings_all = embedding_candidate
                else:
                    embeddings_all = torch.cat((embeddings_all, embedding_candidate), 1)
        else:
            if idx_candidate not in embeddings_idx_model and idx_candidate != -1: 
                embeddings_idx_model.append(idx_candidate)
                if embeddings_all is None:
                    embeddings_all = embedding_candidate
                else:
                    embeddings_all = torch.cat((embeddings_all, embedding_candidate), 1)

    return embeddings_idx_model, embeddings_all


def train_selection(all_embeddings, y_a_i, support_size, n_support, n_way, with_replacement=False):
    embeddings_idx = []
    cross_validation_epoch = 20

    embeddings_best_of_each = []
    embeddings_idx_of_each = []

    for num in range(len(all_embeddings)):
        embedding_candidate = None
        idx_candidate = -1
        min_loss = float("inf")
        for idx, embedding in enumerate(all_embeddings[num]):

            if n_support == 50 or 20:
                running_loss = train_loss_half_validation(embedding, y_a_i, support_size, n_support, n_way, cross_validation_epoch)
            else:
                running_loss = train_loss_cross_validation(embeddings, y_a_i, support_size, n_support, n_way, cross_validation_epoch)

            if running_loss < min_loss:
                embedding_candidate = embedding
                idx_candidate = idx
                min_loss = running_loss
                
        embeddings_idx_of_each.append(idx_candidate)
        embeddings_best_of_each.append(embedding_candidate)


    embeddings_idx_model, embeddings_all = combine_model(embeddings_best_of_each, y_a_i, support_size, n_support, n_way, cross_validation_epoch, with_replacement=with_replacement)
   
    return embeddings_idx_of_each, embeddings_idx_model, embeddings_all, embeddings_best_of_each


def test_loop(novel_loader, return_std = False, loss_type="softmax", n_query = 15, models_to_use=[], finetune_each_model = False, n_way = 5, n_support = 5): #overwrite parrent function
    correct = 0
    count = 0

    iter_num = len(novel_loader) 

    acc_all = []
    for _, (x, y) in enumerate(novel_loader):

        ###############################################################################################
        pretrained_models = []
        for _ in range(len(models_to_use)):
            pretrained_models.append(model_dict[params.model]())

        ###############################################################################################      
        for idx, dataset_name in enumerate(models_to_use):

            checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, models_to_use[idx], params.model, params.method)
            if params.train_aug:
                checkpoint_dir += '_aug'

            params.save_iter = -1
            if params.save_iter != -1:
                modelfile   = get_assigned_file(checkpoint_dir, params.save_iter)
            elif params.method in ['baseline', 'baseline++'] :
                modelfile   = get_resume_file(checkpoint_dir)
            else:
                modelfile   = get_best_file(checkpoint_dir)

            tmp = torch.load(modelfile)
            state = tmp['state']

            state_keys = list(state.keys())
            for _, key in enumerate(state_keys):
                if "feature." in key:
                    newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                    state[newkey] = state.pop(key)
                else:
                    state.pop(key)

            pretrained_models[idx].load_state_dict(state)

        ###############################################################################################
        n_query = x.size(1) - n_support
        x = x.cuda()
        x_var = Variable(x)

        batch_size = 4
        support_size = n_way * n_support 
    
        ##################################################################################
        if finetune_each_model:

            for idx, model_name in enumerate(pretrained_models):
                pretrained_models[idx].cuda()
                pretrained_models[idx].train()

                x_a_i = x_var[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:]) # (25, 3, 224, 224)

                loss_fn = nn.CrossEntropyLoss().cuda()
                cnet = Classifier(pretrained_models[idx].final_feat_dim, n_way).cuda()

                classifier_opt = torch.optim.SGD(cnet.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
                feature_opt = torch.optim.SGD(pretrained_models[idx].parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

                x_a_i = Variable(x_a_i).cuda()
                y_a_i = Variable( torch.from_numpy( np.repeat(range( n_way ), n_support ) )).cuda() # (25,)

                train_size = support_size 
                batch_size = 4
                for epoch in range(100):
                    rand_id = np.random.permutation(train_size)

                    for j in range(0, train_size, batch_size):
                        classifier_opt.zero_grad()
                        feature_opt.zero_grad()

                        #####################################
                        selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, train_size)]).cuda()
                        z_batch = x_a_i[selected_id]

                        y_batch = y_a_i[selected_id] 
                        #####################################
                        outputs = pretrained_models[idx](z_batch)
                        outputs = cnet(outputs)            
                        #####################################
                
                        loss = loss_fn(outputs, y_batch)
                        loss.backward()
                      

                        classifier_opt.step()
                        feature_opt.step()
        

        ###############################################################################################
        for idx, model_name in enumerate(pretrained_models):
            pretrained_models[idx].cuda()
            pretrained_models[idx].eval()

        ###############################################################################################
        
        all_embeddings_train = []

        for idx, model_name in enumerate(pretrained_models):
            model_embeddings = []
            x_a_i = x_var[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:]) # (25, 3, 224, 224)
            for idx, module in enumerate(pretrained_models[idx].trunk):
                x_a_i = module(x_a_i)
                if len(list(x_a_i.size())) == 4:
                    embedding =  F.adaptive_avg_pool2d(x_a_i, (1, 1)).squeeze()
                    model_embeddings.append(embedding.detach())
            
            if params.model == "ResNet10" or params.model == "ResNet18":
                model_embeddings = model_embeddings[4:-1]
                
            elif params.model == "Conv4":
                model_embeddings = model_embeddings

            all_embeddings_train.append(model_embeddings)
     
        ##########################################################
    
        y_a_i = np.repeat(range( n_way ), n_support )
        embeddings_idx_of_each, embeddings_idx_model, embeddings_train, embeddings_best_of_each = train_selection(all_embeddings_train, y_a_i, support_size, n_support, n_way, with_replacement=True)
    
        ##########################################################
        
        all_embeddings_test = []

        for idx, model_name in enumerate(pretrained_models):
            model_embeddings = []

            x_b_i = x_var[:, n_support:,:,:,:].contiguous().view( n_way* n_query,   *x.size()[2:]) 
            for idx, module in enumerate(pretrained_models[idx].trunk):
                x_b_i = module(x_b_i)
                if len(list(x_b_i.size())) == 4:
                    embedding =  F.adaptive_avg_pool2d(x_b_i, (1, 1)).squeeze()
                    model_embeddings .append(embedding.detach())

            if params.model == "ResNet10" or params.model == "ResNet18":
                model_embeddings = model_embeddings[4:-1]

            elif params.model == "Conv4":
                model_embeddings = model_embeddings

            all_embeddings_test.append(model_embeddings)
        
        ############################################################################################
        embeddings_test = []

        for index in embeddings_idx_model:
            embeddings_test.append(all_embeddings_test[index][embeddings_idx_of_each[index]])

        embeddings_test = torch.cat(embeddings_test, 1)
        ############################################################################################


        y_a_i = Variable( torch.from_numpy( np.repeat(range( n_way ), n_support ) )).cuda() # (25,)

        net = Classifier(embeddings_test.size()[1], n_way).cuda()

        loss_fn = nn.CrossEntropyLoss().cuda()
     
        classifier_opt = torch.optim.SGD(net.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)


        total_epoch = 100
        embeddings_train = Variable(embeddings_train.cuda())

        net.train()
        for epoch in range(total_epoch):
            rand_id = np.random.permutation(support_size)

            for j in range(0, support_size, batch_size):
                classifier_opt.zero_grad()

                #####################################
                selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size)]).cuda()
                z_batch = embeddings_train[selected_id]
               
                y_batch = y_a_i[selected_id] 
                #####################################
                outputs = net(z_batch)            
                #####################################
        
                loss = loss_fn(outputs, y_batch)
              
                loss.backward()
                classifier_opt.step()


        embeddings_test = Variable(embeddings_test.cuda())
        
        scores = net(embeddings_test)

        y_query = np.repeat(range( n_way ), n_query )
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        correct_this, count_this = float(top1_correct), len(y_query)
        print (correct_this/ count_this *100)
        acc_all.append((correct_this/ count_this *100))

        ###############################################################################################
        
    acc_all  = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
    
if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')

    ##################################################################
    image_size = 224
    iter_num = 600

    n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot) 
   
    models_to_use = params.models_to_use
    finetune_each_model = params.fine_tune_all_models
    ##################################################################

    dataset_names = ["ISIC", "EuroSAT", "CropDisease", "Chest"]
    novel_loaders = []

    datamgr             =  ISIC_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(aug =False)
    novel_loaders.append(novel_loader)
            
    datamgr             =  EuroSAT_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(aug =False)
    novel_loaders.append(novel_loader)

    datamgr             = CropDisease_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(aug =False)
    novel_loaders.append(novel_loader)

    datamgr             =  Chest_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(aug =False)
    novel_loaders.append(novel_loader)

    #########################################################################

    for idx, novel_loader in enumerate(novel_loaders):
        print dataset_names[idx]
        start_epoch = params.start_epoch
        stop_epoch = params.stop_epoch

        test_loop(novel_loader, return_std = False,  n_query = 15, models_to_use=models_to_use, finetune_each_model = finetune_each_model, **few_shot_params)
