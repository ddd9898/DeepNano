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
from models.models import baseline,DeepNano_seq
from utils.dataloader import seqData_Dscript
from utils.evaluate import evaluate

'''
1. 测试MLP
python train_DScriptData.py --Model 0 --finetune 1 &

2. 测试MLP+Ensemble
CUDA_VISIBLE_DEVICES=1 python train_DScriptData.py --Model 1 --finetune 1  --ESM2 esm2_t6_8M_UR50D  &

CUDA_VISIBLE_DEVICES=2 python train_DScriptData.py --Model 1 --finetune 1  --ESM2 esm2_t12_35M_UR50D  &

CUDA_VISIBLE_DEVICES=3 python train_DScriptData.py --Model 1 --finetune 1  --ESM2 esm2_t30_150M_UR50D  &

CUDA_VISIBLE_DEVICES=2 python train_DScriptData.py --Model 1 --finetune 1  --ESM2 esm2_t33_650M_UR50D  &

CUDA_VISIBLE_DEVICES=3 python train_DScriptData.py --Model 1 --finetune 1  --ESM2 esm2_t36_3B_UR50D  &

CUDA_VISIBLE_DEVICES=3 python train_DScriptData.py --Model 1 --finetune 1  --ESM2 esm2_t48_15B_UR50D  &

'''

def get_args():
    parser = argparse.ArgumentParser(description='Train the model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  

    parser.add_argument('--Model', dest='Model', type=int, default=0,
                        help='Model',metavar='E')
    
    parser.add_argument('--finetune', dest='finetune', type=int, default=1,
                        help='finetune',metavar='E')
    
    parser.add_argument('--ESM2', dest='ESM2', type=str, default=None,
                        help='pretrained',metavar='E')
    

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
        
        if Model_type == 0:
            predictions = model(seqs_nanobody,seqs_antigen,device)

            ###Calculate loss
            gt = data[2].float().to(device)
            loss = F.binary_cross_entropy(predictions.squeeze(),gt)
            
        elif Model_type == 1:
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
            if Model_type == 0:
                predictions = model(seqs_nanobody,seqs_antigen,device)

                total_preds_ave = torch.cat((total_preds_ave, predictions.cpu()), 0)
                
            elif Model_type == 1:
                p_ave, p_min, p_max = model(seqs_nanobody,seqs_antigen,device)
                
                total_preds_ave = torch.cat((total_preds_ave, p_ave.cpu()), 0)
                total_preds_min = torch.cat((total_preds_min, p_min.cpu()), 0)
                total_preds_max = torch.cat((total_preds_max, p_max.cpu()), 0)

            #Ground truth
            g = data[2]
            total_labels = torch.cat((total_labels, g), 0)
            
    if Model_type == 1:
        return total_labels.numpy().flatten(),total_preds_ave.numpy().flatten(),total_preds_min.numpy().flatten(),total_preds_max.numpy().flatten()
    else:
        return total_labels.numpy().flatten(),total_preds_ave.numpy().flatten()



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

    if args.Model == 0:
        model_name = 'baseline'
    elif args.Model == 1:
        model_name = 'DeepNano_seq'

    #Set log
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #Output name
    add_name = '({})_DScriptData_finetune{}'.format(args.ESM2,args.finetune)
    # add_name = '(esm2_t12_35M_UR50D)_DScriptData_finetune{}'.format(args.finetune)
    # add_name = '(esm2_t30_150M_UR50D)_DScriptData_finetune{}'.format(args.finetune)
    
    
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
    

    #Step 1:Prepare dataloader
    trainDataset = seqData_Dscript(pair_path='./data/D_script/pairs/human_train.tsv',seqs_path='./data/D_script/seqs/human_dedup.fasta',addNeg=True,augment=True)
    valDataset = seqData_Dscript(pair_path='./data/D_script/pairs/human_test.tsv',seqs_path='./data/D_script/seqs/human_dedup.fasta',addNeg=True)


    train_loader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True,pin_memory=True)
    val_loader = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=False)


    #Step 2: Set  model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    if args.Model == 0:
        model = baseline(finetune=args.finetune).to(device)
    elif args.Model == 1:
        if args.ESM2 == 'esm2_t6_8M_UR50D':
            model = DeepNano_seq(pretrained_model=r'./models/esm2_t6_8M_UR50D',hidden_size=320, finetune=args.finetune).to(device)
        if args.ESM2 == 'esm2_t12_35M_UR50D':
            model = DeepNano_seq(pretrained_model=r'./models/esm2_t12_35M_UR50D',hidden_size=480,finetune=args.finetune).to(device)
        if args.ESM2 == 'esm2_t30_150M_UR50D':
            model = DeepNano_seq(pretrained_model=r'./models/esm2_t30_150M_UR50D',hidden_size=640,finetune=args.finetune).to(device)
        if args.ESM2 == 'esm2_t33_650M_UR50D':
            model = DeepNano_seq(pretrained_model=r'./models/esm2_t33_650M_UR50D',hidden_size=1280,finetune=args.finetune).to(device)
        if args.ESM2 == 'esm2_t36_3B_UR50D':
            model = DeepNano_seq(pretrained_model=r'./models/esm2_t36_3B_UR50D',hidden_size=2560,finetune=args.finetune).to(device)
        if args.ESM2 == 'esm2_t48_15B_UR50D':
            model = DeepNano_seq(pretrained_model=r'./models/esm2_t48_15B_UR50D',hidden_size=5120,finetune=args.finetune).to(device)

    #Step 3: Train the model
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01) #0.001

                                                    
    logging.info(f'''Starting training:
    Model_name:      {model_name}
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

    early_stop_count = 5
    no_improve_count = 0
    for epoch in range(NUM_EPOCHS):
        #Train
        train_loss = train(model, device, train_loader, optimizer, epoch, args.Model)
        
        #Test
        if args.Model == 1:
            g,p_ave,p_min,p_max = predicting(model, device, val_loader, args.Model)

            #ave
            precision,recall,accuracy,F1_score,Top10,Top20,Top50,AUC_ROC,AUC_PR = evaluate(g,p_ave)
            logging.info("Epoch {} for ave: Top10 = {:.4f},Top20 = {:.4f},Top50 = {:.4f},accuracy={:.4f},Recall = {:.4f},Precision={:.4f},F1 score={:.4f},AUC_ROC={:.4f},AUC_PR={:.4f}".format(
                epoch,Top10,Top20,Top50,accuracy,recall,precision,F1_score,AUC_ROC,AUC_PR))
            #min
            precision,recall,accuracy,F1_score,Top10,Top20,Top50,AUC_ROC,AUC_PR = evaluate(g,p_min)
            logging.info("Epoch {} for min: Top10 = {:.4f},Top20 = {:.4f},Top50 = {:.4f},accuracy={:.4f},Recall = {:.4f},Precision={:.4f},F1 score={:.4f},AUC_ROC={:.4f},AUC_PR={:.4f}".format(
                epoch,Top10,Top20,Top50,accuracy,recall,precision,F1_score,AUC_ROC,AUC_PR))
            #max
            precision,recall,accuracy,F1_score,Top10,Top20,Top50,AUC_ROC,AUC_PR = evaluate(g,p_max)
            logging.info("Epoch {} for max: Top10 = {:.4f},Top20 = {:.4f},Top50 = {:.4f},accuracy={:.4f},Recall = {:.4f},Precision={:.4f},F1 score={:.4f},AUC_ROC={:.4f},AUC_PR={:.4f}".format(
                epoch,Top10,Top20,Top50,accuracy,recall,precision,F1_score,AUC_ROC,AUC_PR))
            #ensemble
            p = (p_ave+p_min+p_max)/3
        else:
            g,p = predicting(model, device, val_loader, args.Model)
        
        precision,recall,accuracy,F1_score,Top10,Top20,Top50,AUC_ROC,AUC_PR = evaluate(g,p)
        logging.info("Epoch {}: Top10 = {:.4f},Top20 = {:.4f},Top50 = {:.4f},accuracy={:.4f},Recall = {:.4f},Precision={:.4f},F1 score={:.4f},AUC_ROC={:.4f},AUC_PR={:.4f}".format(
            epoch,Top10,Top20,Top50,accuracy,recall,precision,F1_score,AUC_ROC,AUC_PR))
        
        
        if best_AUC_PR<AUC_PR:
            best_AUC_PR = AUC_PR
            best_epoch = epoch
            #Save model
            torch.save(model.state_dict(), model_file_name +'_best.model')
            
            no_improve_count = 0
        else:
            no_improve_count = no_improve_count + 1
            
        logging.info("Best epoch {} for ensemble with AUC_PR = {:.4f}".format(best_epoch,best_AUC_PR))
        
        ##Early stop
        if no_improve_count==early_stop_count:
            logging.info("Early stop!")
            break



            

        