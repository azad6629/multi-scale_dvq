import yaml
import os
import matplotlib.pyplot as plt
from datetime import datetime
import time
import argparse
import random
import numpy as np
import pandas as pd

import torch
import torchvision
import torch.optim as optim
import torch.backends.cudnn as cudnn
from data import DataLoader
from utils import log
import scipy.stats as stats
from data.constants import NIH_TASKS, NIH_CXR8_TASKS
from utils import log, computeAUROC, computeF1, computePrecision_Recall
from dynamic_chex_quant_models import dynamic_dense121

   

import warnings
warnings.filterwarnings("ignore")
from fastprogress import master_bar, progress_bar


models = ['dynamic_dense121']

parser = argparse.ArgumentParser(description='PyTorch Medical Image Supervised Training.')
parser.add_argument('-model_name', help='choice of model, choices=model_names')
parser.add_argument('-epoch', default=300, help='total epoches.')
parser.add_argument('-batch', default=64, help='total epoches.')
parser.add_argument('-dataset', default='nih', help='choice of dataset.')
parser.add_argument('-task', default='nih_tasks', help='(nih_tasks')
parser.add_argument('-resume', default=False, action='store_true', help='To resume the pre-training.')
parser.add_argument('-ne', type=int, default=1024, help='number of codebook vectors.')
parser.add_argument('-ed', type=int, default=64, help='dimension of codebook vectors.')
parser.add_argument('-cc', type=float, default=0.25, help='commitment cost.')
parser.add_argument('-seed', default = 42, help='seed for initializing training. ')
parser.add_argument('-gpu', default=0,help='GPU id to use.')

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def train(model,criterion,optimizer,loader, device, args):
    model.train()
    losstrain = 0
    for idx, (img,target) in enumerate(progress_bar(loader)):
        target = target.to(device)
        img = img.to(device)
        output, qloss  = model(img)
        closs = criterion(output, target)
        loss = 10*closs + qloss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losstrain += loss.item()
    return losstrain / len(loader)

def valid(model,loader,device, args):
    cudnn.benchmark = True
    outGT = torch.FloatTensor().to(device)
    outPRED = torch.FloatTensor().to(device)
    model.eval()
    with torch.no_grad():
        for idx, (img,target) in enumerate(progress_bar(loader)):
            target = target.to(device)
            img = img.to(device)
            outGT = torch.cat((outGT, target),0)
            outGT = outGT.to(device)
            output, _  = model(img)
            
            outPRED = torch.cat((outPRED,output), 0)
    aurocIndividual = computeAUROC(outGT, outPRED)
    aurocMean = np.array(aurocIndividual).mean()
    f1        = np.nanmean(np.array(computeF1(outGT, outPRED)))
    p,r       = computePrecision_Recall(outGT, outPRED)
    precision = np.nanmean(p)
    recall    = np.nanmean(r)
    return aurocIndividual, aurocMean, f1, precision, recall

def test(model,loader,device, num_classes,args):
    cudnn.benchmark = True
    outGT = torch.FloatTensor().to(device)
    outPRED = torch.FloatTensor().to(device)
    model.eval()
    confidences = []  # List to store confidence values for each sample
    class_confidences = {i: [] for i in range(num_classes)}  # Dictionary to store confidence values for each class


    with torch.no_grad():
        for idx, (img,target) in enumerate(progress_bar(loader)):
            target = target.to(device)
            img = img.to(device)
            outGT = torch.cat((outGT, target),0)
            outGT = outGT.to(device)
            output, _  = model(img)
            outPRED = torch.cat((outPRED,output), 0)
            
    aurocIndividual = computeAUROC(outGT, outPRED)
    aurocMean = np.array(aurocIndividual).mean()
    f1        = np.nanmean(np.array(computeF1(outGT, outPRED)))
    p,r       = computePrecision_Recall(outGT, outPRED)
    precision = np.nanmean(p)
    recall    = np.nanmean(r)
    
    return outGT,outPRED, aurocIndividual, aurocMean, f1, precision, recall


def main():
    config_file = "config.yaml"
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        
    args = parser.parse_args() 
    
    if args.seed is not None:
        seed = args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    
    config['model']= args.model_name 
    print(f"Training model: {args.model_name}")
    config['dataset']= args.dataset
    config['task']= args.task
    config['vqconfig']['codebook_size']  = args.ne
    config['vqconfig']['codebook_dim']   = args.ed
    config['vqconfig']['commitment_beta']= args.cc
    resume = args.resume    
    config['train_bs'] = int(args.batch)

    data_ins   = DataLoader(config)
    train_loader,valid_loader, test_loader = data_ins.GetNihDataset()

    num_classes = 15
    class_names = NIH_TASKS

    file_name = f'./dynamic_quant_ckpt/DYNAMIC_ne-{args.ne}_ed-{args.ed}_cc-{args.cc}_bs-{args.batch}'
    save_path   = os.path.join(file_name,args.model_name.lower())

    method_name = f"{config['model']}_{config['dataset']}_{config['total_epochs']}"
    config['ckpt_path'] = os.path.join(save_path,method_name)        
    os.makedirs(config['ckpt_path'], exist_ok=True)
    logger = log(path=config['ckpt_path'], file=f"{method_name}.logs")    
    logger.info('Trained with dynamic codebook')

    model_class = globals()[args.model_name]
    model = model_class(config).to(device)

    num_params = count_parameters(model)
    log_step   = config['log_step']
    now = datetime.now()
    current_time = now.strftime("%d/%m/%Y %H:%M:%S")

    logger.info(current_time)
    logger.info('Training data Info:')
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    logger.info(config)
    logger.info(f"dataset is {config['dataset']}") 
    logger.info(model)

    optimizer=optim.Adam(model.parameters(),lr=float(config['lr']),betas=(0.9, 0.999),eps = 1e-08,weight_decay =float(config['wd']))

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = float(config['schedule_factor']),
                                  patience = float(config['schedule_patience']), mode = 'max', verbose=True)
    criterion = torch.nn.BCELoss()

    if resume:
        checkpoint_path = os.path.join(config['ckpt_path'], f"{method_name}.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            start_epoch = checkpoint['epoch']
            best_auc = checkpoint['best_auc']
            model.load_state_dict(checkpoint['model'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info(f"--> Loaded checkpoint '{checkpoint_path}' (epoch {start_epoch})")
    else:
        start_epoch = 0
        best_auc = 0.0
        logger.info("--> No loaded checkpoint!")


    count=0
    best_epoch = 0
    patience= config['patience']
    training_losses   = []
    validation_auc    = []
    validation_f1     = []
    validation_PR     = []
    validation_RC     = []

    mb = master_bar(range(config['total_epochs']))
    mb.names = ['Train loss', 'AUROC','f1_score', 'PR_score', 'RC_score']
    start_time = time.time()
    x = []
    out_path = os.path.join(config['ckpt_path'], f'{method_name}.csv')

    for epoch in mb:
        loss = train(model,criterion,optimizer,train_loader,device,args)
        aurocIndividual,aurocMean, f1 , precision, recall= valid(model,valid_loader,device,args)
        training_losses.append(loss)
        validation_auc.append(aurocMean)
        validation_f1.append(f1)
        validation_PR.append(precision)
        validation_RC.append(recall)

        logger.info(f'Epoch: [{epoch}]\t'
                    f'Train_loss {loss: .5f}\t'
                    f'AUC {aurocMean:.3f}\t'
                    f'F1 {f1:.2f}\t'
                    f'PR {precision:.3f}\t'
                    f'RC {recall:.3f}\t'
                   )
        scheduler.step(aurocMean)
        # Update training chart
        mb.update_graph([[x, training_losses], [x, aurocMean], [x, validation_f1],
                         [x, validation_PR],[x, validation_RC]],[0,epoch+1+round(epoch*0.3)], [0,1])

        model_state = {'config': config,
                       'epoch': epoch,
                       'best_auc':best_auc,
                       'model': model.state_dict(),
                       'optimizer': optimizer.state_dict(),
                 }

        SAVE_PATH1 = os.path.join(config['ckpt_path'], f'{method_name}.pth')
        torch.save(model_state, SAVE_PATH1)

        if aurocMean > best_auc:
            logger.info('auc increased ({:.4f} --> {:.4f}). Saving model ...'.format(best_auc,aurocMean))
            SAVE_PATH2 = os.path.join(config['ckpt_path'], f'{method_name}_best.pth')
            torch.save(model_state, SAVE_PATH2)
            best_auc = aurocMean
            best_epoch = epoch
            count = 0  
        else:
            count += 1
        if count >= patience:
            logger.info(f"No improvement for {patience} epochs. Early stopping...")
            break

    logger.info(f"Best AUC: {best_auc:.3f} at epoch {best_epoch}") 

    for i in range (0, len(aurocIndividual)):
        logger.info(f'{class_names[i]}\t\t{aurocIndividual[i]:.3f}')

    #Testing
    checkpoint_path = os.path.join(config['ckpt_path'], f"{method_name}_best.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model = model.to(device)
    outGT,outPRED, aurocIndividual, aurocMean, f1 , precision, recall = test(model,test_loader,device,num_classes,args)

    save_path = os.path.join(config['ckpt_path'] )
    plot_conf(outGT,outPRED,args.model_name,class_names,save_path)

    logger.info(f'\nTest Results: \t'
                f'AUC {aurocMean:.3f}\t'
                f'F1 {f1:.2f}\t'
                f'PR {precision:.3f}\t'
                f'RC {recall:.3f}\t\n'

                )
    for i in range (0, len(aurocIndividual)):
        logger.info(f'{class_names[i]}\t\t{aurocIndividual[i]:.3f}\t\t{average_class_confidences[i]:.3f}')
        
    # Reset for the next model
    del model
    torch.cuda.empty_cache()
              
if __name__ == '__main__':
    main()