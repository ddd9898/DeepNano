import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse
import logging
# from torch.utils.tensorboard import SummaryWriter
import random
import os
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")
from models.models import DeepNano_seq
from utils.dataloader import seqData_NBAT
from utils.evaluate import evaluate


'''
与论文Sequence-Based Nanobody-Antigen Binding Predictionn中的算法比较,在其整理的数据上训练模型

CUDA_VISIBLE_DEVICES=1 python train_NBAT_Data.py --Model 0 --fold 0
CUDA_VISIBLE_DEVICES=1 python train_NBAT_Data.py --Model 0 --fold 1
CUDA_VISIBLE_DEVICES=1 python train_NBAT_Data.py --Model 0 --fold 2 
CUDA_VISIBLE_DEVICES=1 python train_NBAT_Data.py --Model 0 --fold 3 
CUDA_VISIBLE_DEVICES=1 python train_NBAT_Data.py --Model 0 --fold 4 

'''

def get_args():
    parser = argparse.ArgumentParser(description='Train the model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 
    parser.add_argument('--Model', dest='Model', type=int, default=0,
                        help='Model',metavar='E')
    
    parser.add_argument('--fold', dest='fold', type=int, default=0,
                        help='Number of fold',metavar='E')

    return parser.parse_args()



def train(model, device, train_loader, optimizer, epoch, Model_type):
    '''
    training function at each epoch
    '''
    # print('Training on {} samples...'.format(len(train_loader.dataset)))
    logging.info('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        #Get input
        seqs_nanobody = data[0]
        seqs_antigen = data[1]
        
        #Calculate output
        optimizer.zero_grad()
        p_ave, p_min, p_max = model(seqs_nanobody,seqs_antigen,device)
        
        ###Calculate loss
        gt = data[2].float().to(device)
        loss1 = F.binary_cross_entropy(p_ave.squeeze(),gt)
        loss2 = F.binary_cross_entropy(p_min.squeeze(),gt)
        loss3 = F.binary_cross_entropy(p_max.squeeze(),gt)
        
        loss = (loss1 + loss2 + loss3)/3

    
        train_loss = train_loss + loss.item()

        #Optimize the model
        loss.backward()
        optimizer.step()

        if batch_idx % LOG_INTERVAL == 0:
            logging.info('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                            batch_idx * BATCH_SIZE,
                                                                            len(train_loader.dataset),
                                                                            100. * batch_idx / len(train_loader),
                                                                            loss.item()))
    train_loss = train_loss / len(train_loader)
    return train_loss


def predicting(model, device, loader, Model_type):
    model.eval()
    total_preds_ave = torch.Tensor()
    total_preds_min = torch.Tensor()
    total_preds_max = torch.Tensor()
    total_labels = torch.Tensor()

    logging.info('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        # for data in tqdm(loader):
        for data in loader:
            #Get input
            seqs_nanobody = data[0]
            seqs_antigen = data[1]

            #Calculate output
            p_ave, p_min, p_max = model(seqs_nanobody,seqs_antigen,device)
            
            total_preds_ave = torch.cat((total_preds_ave, p_ave.cpu()), 0)
            total_preds_min = torch.cat((total_preds_min, p_min.cpu()), 0)
            total_preds_max = torch.cat((total_preds_max, p_max.cpu()), 0)
        
            g = data[2]
            total_labels = torch.cat((total_labels, g), 0)

    return total_labels.numpy().flatten(),total_preds_ave.numpy().flatten(),total_preds_min.numpy().flatten(),total_preds_max.numpy().flatten()







def set_seed(seed = 1998):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministics = True


if __name__ == '__main__':
    
    set_seed()

    #Train setting
    BATCH_SIZE = 32
    LR = 0.00005
    LOG_INTERVAL = 20000
    NUM_EPOCHS = 10

    #Get argument parse
    args = get_args()

    #Set log
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #Output name
    if args.Model == 0:
        model_name = 'DeepNano_seq'

        
        
    add_name = '(esm2_t6_8M_UR50D)_NBATdata_fold{}_finetune1'.format(args.fold)
    
    
    logfile = './output/log/log_' + model_name + add_name + '.txt'
    fh = logging.FileHandler(logfile,mode='a')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # #Tensorboard
    # logfile = './output/log/log_' + model_name + add_name
    # writer = SummaryWriter(logfile)

    #Step 1:Prepare dataloader
    trainDataset = seqData_NBAT(pair_path='./data/Nanobody_Antigen-main/all_pair_data.csv',
                        split_path='./data/Nanobody_Antigen-main/all_test_split.csv',
                        data_split='{}-train'.format(args.fold),
                        addNeg=True)
    valDataset = seqData_NBAT(pair_path='./data/Nanobody_Antigen-main/all_pair_data.csv',
                        split_path='./data/Nanobody_Antigen-main/all_test_split.csv',
                        data_split='{}-test'.format(args.fold),
                        addNeg=True)
    # print("Train size = {}".format(len(trainDataset)))
    # print("Validation size = {}".format(len(valDataset)))

   
    train_loader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=False)


    #Step 2: Set  model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    if args.Model == 0:
        model = DeepNano_seq(finetune=1).to(device)

    # model = torch.nn.DataParallel(model)

    #Step 3: Train the model
    # optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01) #0.001
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)

                                                    
    logging.info(f'''Starting training:
    Epochs:          {NUM_EPOCHS}
    Batch size:      {BATCH_SIZE}
    Learning rate:   {LR}
    Training size:   {len(trainDataset)}
    Validating size: {len(valDataset)}
    Device:          {device.type}
    ''')
    
    best_AUC_PR = -1
    best_epoch = 0
    model_file_name =  './output/checkpoint/' + model_name + add_name

    for epoch in range(NUM_EPOCHS):
        #Train
        train_loss = train(model, device, train_loader, optimizer, epoch, args.Model)
        
        #Test
        g,p_ave,p_min,p_max = predicting(model, device, val_loader, args.Model)
        p = (p_ave+p_min+p_max)/3
        
        precision,recall,accuracy,F1_score,Top10,Top20,Top50,AUC_ROC,AUC_PR = evaluate(g,p)
        logging.info("Epoch {}: Top10 = {:.4f},Top20 = {:.4f},Top50 = {:.4f},accuracy={:.4f},Recall = {:.4f},Precision={:.4f},F1 score={:.4f},AUC_ROC={:.4f},AUC_PR={:.4f}".format(
            epoch,Top10,Top20,Top50,accuracy,recall,precision,F1_score,AUC_ROC,AUC_PR))
        
        
        if best_AUC_PR<AUC_PR:
            best_AUC_PR = AUC_PR
            best_epoch = epoch
            #Save model
            torch.save(model.state_dict(), model_file_name +'_best.model')
        logging.info("Best epoch {} for ensemble with AUC_PR = {:.4f}".format(best_epoch,best_AUC_PR))

            


            

        

        







