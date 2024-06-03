import torch
from torch import nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch.nn import functional as F


class baseline(nn.Module):
    def __init__(self,pretrained_model=r'./models/esm2_t6_8M_UR50D',hidden_size=320,finetune=1):
        super(baseline, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.pretrained_model  = AutoModel.from_pretrained(pretrained_model)

        if finetune == 0: # 冻结pretrained_model参数
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
        elif finetune == 1: # 微调pretrained_model最后一层参数
            for name,param in self.pretrained_model.named_parameters():
                if 'esm2_t6_8M_UR50D' in pretrained_model and 'encoder.layer.5.' not in name:
                    param.requires_grad = False
                if 'esm2_t12_35M_UR50D' in pretrained_model and 'encoder.layer.11.' not in name:
                    param.requires_grad = False
                if 'esm2_t30_150M_UR50D' in pretrained_model and 'encoder.layer.29.' not in name:
                    param.requires_grad = False
                if 'esm2_t33_650M_UR50D' in pretrained_model and 'encoder.layer.32.' not in name:
                    param.requires_grad = False
                if 'esm2_t36_3B_UR50D' in pretrained_model and 'encoder.layer.35.' not in name:
                    param.requires_grad = False
                if 'esm2_t48_15B_UR50D' in pretrained_model and 'encoder.layer.47.' not in name:
                    param.requires_grad = False
        elif finetune == 2: # 微调pretrained_model全部参数
            for param in self.pretrained_model.parameters():
                param.requires_grad = True
                
                
        self.predict_module =   nn.Sequential(
                                nn.Linear(hidden_size*2, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
                                nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                nn.Linear(256, 1),nn.Sigmoid())


    def forward(self,seq1,seq2,device):
        
        #Nanobody
        tokenizer1 = self.tokenizer(seq1,
                                    truncation=True,
                                    padding=True,
                                    max_length=800,
                                    add_special_tokens=False)
        input1_ids=torch.tensor(tokenizer1['input_ids']).to(device)
        attention_mask1=torch.tensor(tokenizer1['attention_mask']).to(device)
        temp_output1=self.pretrained_model(input_ids=input1_ids,attention_mask=attention_mask1) 
        feature_seq1_ave = torch.mean(temp_output1.last_hidden_state,dim=1)
     
        #Antigen
        tokenizer2 = self.tokenizer(seq2,
                                    truncation=True,
                                    padding=True,
                                    max_length=800,
                                    add_special_tokens=False)
        input_ids2=torch.tensor(tokenizer2['input_ids']).to(device)
        attention_mask2=torch.tensor(tokenizer2['attention_mask']).to(device)
        temp_output2=self.pretrained_model(input_ids=input_ids2,attention_mask=attention_mask2)
        feature_seq2_ave = torch.mean(temp_output2.last_hidden_state,dim=1)
        
        
        ##MLP
        feature_seq = torch.cat((feature_seq1_ave,feature_seq2_ave),dim=1)
        prediction = self.predict_module(feature_seq)
        
        #Output
        return prediction

class DeepNano_seq(nn.Module):
    def __init__(self,pretrained_model=r'./models/esm2_t6_8M_UR50D',hidden_size=320,finetune=1):
        super(DeepNano_seq, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.pretrained_model  = AutoModel.from_pretrained(pretrained_model)

        if finetune == 0: # 冻结pretrained_model参数
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
        elif finetune == 1: # 微调pretrained_model最后一层参数
            for name,param in self.pretrained_model.named_parameters():
                if 'esm2_t6_8M_UR50D' in pretrained_model and 'encoder.layer.5.' not in name:
                    param.requires_grad = False
                if 'esm2_t12_35M_UR50D' in pretrained_model and 'encoder.layer.11.' not in name:
                    param.requires_grad = False
                if 'esm2_t30_150M_UR50D' in pretrained_model and 'encoder.layer.29.' not in name:
                    param.requires_grad = False
                if 'esm2_t33_650M_UR50D' in pretrained_model and 'encoder.layer.32.' not in name:
                    param.requires_grad = False
                if 'esm2_t36_3B_UR50D' in pretrained_model and 'encoder.layer.35.' not in name:
                    param.requires_grad = False
                if 'esm2_t48_15B_UR50D' in pretrained_model and 'encoder.layer.47.' not in name:
                    param.requires_grad = False
        elif finetune == 2: # 微调pretrained_model全部参数
            for param in self.pretrained_model.parameters():
                param.requires_grad = True
        
        self.predict_module_ave  =  nn.Sequential(
                                    nn.Linear(hidden_size*2, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
                                    nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                    nn.Linear(256, 1),nn.Sigmoid())
        self.predict_module_max  =  nn.Sequential(
                                    nn.Linear(hidden_size*2, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
                                    nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                    nn.Linear(256, 1),nn.Sigmoid())
        self.predict_module_min  =  nn.Sequential(
                                    nn.Linear(hidden_size*2, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
                                    nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                    nn.Linear(256, 1),nn.Sigmoid())
       

    def forward(self,seq1,seq2,device):
        
        #Nanobody
        tokenizer1 = self.tokenizer(seq1,
                                    truncation=True,
                                    padding=True,
                                    max_length=800,
                                    add_special_tokens=False)
        input1_ids=torch.tensor(tokenizer1['input_ids']).to(device)
        attention_mask1=torch.tensor(tokenizer1['attention_mask']).to(device)
        temp_output1=self.pretrained_model(input_ids=input1_ids,attention_mask=attention_mask1) 
        feature_seq1_ave = torch.mean(temp_output1.last_hidden_state,dim=1)
        feature_seq1_max = torch.max(temp_output1.last_hidden_state,dim=1).values
        feature_seq1_min = torch.min(temp_output1.last_hidden_state,dim=1).values
     
        #Antigen
        tokenizer2 = self.tokenizer(seq2,
                                    truncation=True,
                                    padding=True,
                                    max_length=800,
                                    add_special_tokens=False)
        input_ids2=torch.tensor(tokenizer2['input_ids']).to(device)
        attention_mask2=torch.tensor(tokenizer2['attention_mask']).to(device)
        temp_output2=self.pretrained_model(input_ids=input_ids2,attention_mask=attention_mask2)
        feature_seq2_ave = torch.mean(temp_output2.last_hidden_state,dim=1)
        feature_seq2_max = torch.max(temp_output2.last_hidden_state,dim=1).values
        feature_seq2_min = torch.min(temp_output2.last_hidden_state,dim=1).values
        
        
        ##MLP
        feature_seq_ave = torch.cat((feature_seq1_ave,feature_seq2_ave),dim=1)
        p_ave = self.predict_module_ave(feature_seq_ave)
        
        feature_seq_min = torch.cat((feature_seq1_min,feature_seq2_min),dim=1)
        p_min = self.predict_module_min(feature_seq_min)
        
        feature_seq_max = torch.cat((feature_seq1_max,feature_seq2_max),dim=1)
        p_max = self.predict_module_max(feature_seq_max)
        
        #Output
        return p_ave, p_min, p_max



class Residual_Units(nn.Module):
    def __init__(self, dim_input, dim_hidden):
        super(Residual_Units, self).__init__()
        self.layer1 = nn.Linear(dim_input, dim_hidden)
        self.layer2 = nn.Linear(dim_hidden, dim_input)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = inputs
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        outputs = self.relu(x + inputs)
        return outputs

        
class DeepNano_site(nn.Module):
    def __init__(self,pretrained_model=r'./models/esm2_t6_8M_UR50D',hidden_size=320,finetune=1):
        super(DeepNano_site, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.pretrained_model  = AutoModel.from_pretrained(pretrained_model)

        if finetune == 0: # 冻结pretrained_model参数
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
        elif finetune == 1: # 微调pretrained_model最后一层参数
            for name,param in self.pretrained_model.named_parameters():
                if 'esm2_t6_8M_UR50D' in pretrained_model and 'encoder.layer.5.' not in name:
                    param.requires_grad = False
                if 'esm2_t12_35M_UR50D' in pretrained_model and 'encoder.layer.11.' not in name:
                    param.requires_grad = False
                if 'esm2_t30_150M_UR50D' in pretrained_model and 'encoder.layer.29.' not in name:
                    param.requires_grad = False
                if 'esm2_t33_650M_UR50D' in pretrained_model and 'encoder.layer.32.' not in name:
                    param.requires_grad = False
                if 'esm2_t36_3B_UR50D' in pretrained_model and 'encoder.layer.35.' not in name:
                    param.requires_grad = False
                if 'esm2_t48_15B_UR50D' in pretrained_model and 'encoder.layer.47.' not in name:
                    param.requires_grad = False
        elif finetune == 2: # 微调pretrained_model全部参数
            for param in self.pretrained_model.parameters():
                param.requires_grad = True
        
        ##Prediction module
        self.predict_module = nn.Sequential(Residual_Units(hidden_size*2,1024),
                                            Residual_Units(hidden_size*2,1024),
                                            Residual_Units(hidden_size*2,1024),
                                            nn.Linear(hidden_size*2, 1),nn.Dropout(p=0.4),nn.Sigmoid())

    def forward(self,seq1,seq2,device):
        #Nanobody
        tokenizer1 = self.tokenizer(seq1,
                                    truncation=True,
                                    padding=True,
                                    max_length=800,
                                    add_special_tokens=False)
        input1_ids=torch.tensor(tokenizer1['input_ids']).to(device)
        attention_mask1=torch.tensor(tokenizer1['attention_mask']).to(device)
        temp_output1=self.pretrained_model(input_ids=input1_ids,attention_mask=attention_mask1) 
        feature_seq1_ave = torch.mean(temp_output1.last_hidden_state,dim=1)   
     
        #Antigen
        tokenizer2 = self.tokenizer(seq2,
                                    truncation=True,
                                    padding=True,
                                    max_length=800,
                                    add_special_tokens=False)
        input_ids2=torch.tensor(tokenizer2['input_ids']).to(device)
        attention_mask2=torch.tensor(tokenizer2['attention_mask']).to(device)
        temp_output2=self.pretrained_model(input_ids=input_ids2,attention_mask=attention_mask2)
        feature_seq2_ave = temp_output2.last_hidden_state
        
        ##Concat
        feature_seq1_ave = feature_seq1_ave.unsqueeze(1).repeat(1, feature_seq2_ave.shape[1], 1)
        feature_seq = torch.cat((feature_seq1_ave,feature_seq2_ave),dim=2)
        
        
        pre = self.predict_module(feature_seq)
        
        
        return pre


class DeepNano(nn.Module):
    def __init__(self,pretrained_model=r'./models/esm2_t6_8M_UR50D',hidden_size=320,d_prompt=8,finetune=1,
                 Model_BSite_path=r'./output/checkpoint/DeepNano_site(esm2_t6_8M_UR50D)_SabdabData_finetune1_TF0_best.model'):
        super(DeepNano, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.pretrained_model  = AutoModel.from_pretrained(pretrained_model)

        if finetune == 0: # 冻结pretrained_model参数
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
        elif finetune == 1: # 微调pretrained_model最后一层参数
            for name,param in self.pretrained_model.named_parameters():
                if 'esm2_t6_8M_UR50D' in pretrained_model and 'encoder.layer.5.' not in name:
                    param.requires_grad = False
                if 'esm2_t12_35M_UR50D' in pretrained_model and 'encoder.layer.11.' not in name:
                    param.requires_grad = False
                if 'esm2_t30_150M_UR50D' in pretrained_model and 'encoder.layer.29.' not in name:
                    param.requires_grad = False
                if 'esm2_t33_650M_UR50D' in pretrained_model and 'encoder.layer.32.' not in name:
                    param.requires_grad = False
                if 'esm2_t36_3B_UR50D' in pretrained_model and 'encoder.layer.35.' not in name:
                    param.requires_grad = False
                if 'esm2_t48_15B_UR50D' in pretrained_model and 'encoder.layer.47.' not in name:
                    param.requires_grad = False
        elif finetune == 2: # 微调pretrained_model全部参数
            for param in self.pretrained_model.parameters():
                param.requires_grad = True

        ##Antigen_BSite_predictor
        self.Antigen_BSite_predictor = DeepNano_site(pretrained_model=pretrained_model,hidden_size=hidden_size,finetune=0)
        # model_path = './output/checkpoint/DeepNano_site(esm2_t6_8M_UR50D)_SabdabData_finetune1_TF0_best.model' 
        weights = torch.load(Model_BSite_path,map_location=torch.device('cpu')) # map_location=torch.device('cpu')
        self.Antigen_BSite_predictor.load_state_dict(weights)
        for param in self.Antigen_BSite_predictor.parameters():
            param.requires_grad = False
        
        ###Prompt encoder for Binding sites of binder protein
        self.embeddingLayer = nn.Embedding(2, d_prompt)
        self.positionalEncodings = nn.Parameter(torch.rand(4000, d_prompt), requires_grad=True)

        encoder_layers = nn.TransformerEncoderLayer(d_prompt, nhead=8,dim_feedforward=128,dropout=0.4)
        encoder_norm = nn.LayerNorm(d_prompt)
        self.Prompt_encoder = nn.TransformerEncoder(encoder_layers,1,encoder_norm) #2
        if finetune == 0: # 冻结Prompt_encoder、embeddingLayer、positionalEncodings参数
            for param in self.embeddingLayer.parameters():
                param.requires_grad = False
            self.positionalEncodings.requires_grad = False
            for param in self.Prompt_encoder.parameters():
                param.requires_grad = False
        
        self.predict_module_ave  =  nn.Sequential(
                                    nn.Linear(hidden_size*2+d_prompt, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
                                    nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                    nn.Linear(256, 1),nn.Sigmoid())
        self.predict_module_max  =  nn.Sequential(
                                    nn.Linear(hidden_size*2+d_prompt, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
                                    nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                    nn.Linear(256, 1),nn.Sigmoid())
        self.predict_module_min  =  nn.Sequential(
                                    nn.Linear(hidden_size*2+d_prompt, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
                                    nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                    nn.Linear(256, 1),nn.Sigmoid())

    def forward(self,seq1,seq2,device):
        
        #Nanobody
        tokenizer1 = self.tokenizer(seq1,
                                    truncation=True,
                                    padding=True,
                                    max_length=800,
                                    add_special_tokens=False)
        input1_ids=torch.tensor(tokenizer1['input_ids']).to(device)
        attention_mask1=torch.tensor(tokenizer1['attention_mask']).to(device)
        temp_output1=self.pretrained_model(input_ids=input1_ids,attention_mask=attention_mask1) 
        feature_seq1_ave = torch.mean(temp_output1.last_hidden_state,dim=1)
        feature_seq1_max = torch.max(temp_output1.last_hidden_state,dim=1).values
        feature_seq1_min = torch.min(temp_output1.last_hidden_state,dim=1).values
     
        #Antigen
        tokenizer2 = self.tokenizer(seq2,
                                    truncation=True,
                                    padding=True,
                                    max_length=800,
                                    add_special_tokens=False)
        input_ids2=torch.tensor(tokenizer2['input_ids']).to(device)
        attention_mask2=torch.tensor(tokenizer2['attention_mask']).to(device)
        temp_output2=self.pretrained_model(input_ids=input_ids2,attention_mask=attention_mask2)
        feature_seq2_ave = torch.mean(temp_output2.last_hidden_state,dim=1)
        feature_seq2_max = torch.max(temp_output2.last_hidden_state,dim=1).values
        feature_seq2_min = torch.min(temp_output2.last_hidden_state,dim=1).values

        ##Prompt
        BSite2 = self.Antigen_BSite_predictor(seq1,seq2,device)
        BSite2 = (BSite2.squeeze()>0.5)+0
        BSite2_embedding = self.embeddingLayer(BSite2)  #batch * seq * feature
        BSite2_embedding = BSite2_embedding + self.positionalEncodings[:BSite2_embedding.shape[1],:]

        BSite2_embedding = BSite2_embedding.permute(1,0,2) #seq * batch * feature
        BSite2_embedding = self.Prompt_encoder(BSite2_embedding)
        BSite2_embedding = BSite2_embedding.permute(1,0,2) #batch * seq * feature

        BSite2_embedding_ave = torch.mean(BSite2_embedding,dim = 1)
        
        ##Add
        feature_seq2_ave = torch.cat((feature_seq2_ave,BSite2_embedding_ave),dim=1)
        feature_seq2_max = torch.cat((feature_seq2_max,BSite2_embedding_ave),dim=1)
        feature_seq2_min = torch.cat((feature_seq2_min,BSite2_embedding_ave),dim=1)
        
        
        ##MLP
        feature_seq_ave = torch.cat((feature_seq1_ave,feature_seq2_ave),dim=1)
        p_ave = self.predict_module_ave(feature_seq_ave)
        
        feature_seq_min = torch.cat((feature_seq1_min,feature_seq2_min),dim=1)
        p_min = self.predict_module_min(feature_seq_min)
        
        feature_seq_max = torch.cat((feature_seq1_max,feature_seq2_max),dim=1)
        p_max = self.predict_module_max(feature_seq_max)
        
        #Output
        return p_ave, p_min, p_max

