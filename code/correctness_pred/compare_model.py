import pickle
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import random

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score

from prettytable import PrettyTable

from scipy.stats.distributions import chi2

class BertModel(nn.Module):
    def __init__(self, sentence_dim, skill_dim, dropout):
        super().__init__()
        self.skill_dim = skill_dim
        self.fc_test = nn.Linear(768,sentence_dim)
        self.fc_question = nn.Linear(768,sentence_dim)
        self.fc_answer = nn.Linear(768,sentence_dim)
        self.fc_skill = nn.Linear(skill_dim,skill_dim*2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(3*sentence_dim+skill_dim*2,128)
#         self.fc2 = nn.Linear(sentence_dim+skill_dim*2,128)
        self.out = nn.Linear(128,1)
    
    def forward(self, skills,test,question,answer):
        x1 = self.fc_skill(skills[:,:self.skill_dim])
        x2 = self.fc_test(test)
        x3 = self.fc_question(question)
        x4 = self.fc_answer(answer)
        x = torch.cat((x1,x2,x3,x4),dim=1)
#         x = torch.cat((x1,x2),dim=1)
        x = self.fc2(self.relu(x))
        pred = self.out(self.relu(x))
        
        return pred

class SimpleNet(nn.Module):
    def __init__(self, skill_dim):
        super().__init__()
        self.skill_dim = skill_dim
        self.fc_skill = nn.Linear(skill_dim,skill_dim*2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(skill_dim*2,128)
        self.out = nn.Linear(128,1)
        
    def forward(self, skills):
        x1 = self.fc_skill(skills[:,:self.skill_dim])
        x = self.fc2(self.relu(x1))
        pred = self.out(self.relu(x))
        return pred

def create_tensors(data_list):
    # create tensor that is compatible to load and train in the language model
    ds = {}
    keys = ['skills','subtests','questions','answers','y']
    for key in keys:
        ds[key] = []
    
    for entry in data_list:
        ds['skills'].append(entry[0])
        ds['subtests'].append(entry[1])
        ds['questions'].append(entry[2])
        ds['answers'].append(entry[3])
        ds['y'].append(entry[4])
    
    ds['skills'] = torch.tensor(ds['skills']).type(torch.float)
    ds['subtests'] = torch.tensor(ds['subtests'])
    ds['questions'] = torch.tensor(ds['questions'])
    ds['answers'] = torch.tensor(ds['answers'])
    ds['y'] = torch.tensor(ds['y']).type(torch.float)

    return ds

def McNemar_test(pred_1,pred_2,y):
    rounded_preds_1 = torch.round(torch.sigmoid(pred_1))
    correct_1 = rounded_preds_1 == y
    
    rounded_preds_2 = torch.round(torch.sigmoid(pred_2))
    correct_2 = rounded_preds_2 == y
    
    tl = 0
    tr = 0
    bl = 0
    br = 0
    for i in range(len(correct_1)):
        if correct_1[i] == 1 and correct_2[i] == 1:
            tl += 1
        elif correct_1[i] == 1 and correct_2[i] == 0:
            tr += 1
        elif correct_1[i] == 0 and correct_2[i] == 1:
            bl += 1
        elif correct_1[i] == 0 and correct_2[i] == 0:
            br += 1
        else:
            print('Unexpected value in counting correctness',correct_1[i],correct_2[i])
    
    t = PrettyTable(['', 'Model 2 Correct', 'Model 2 Wrong'])
    t.add_row(['Model 1 Correct', tl, tr])
    t.add_row(['Model 1 Wrong', bl, br])
    print(t)
    return tr,bl

def main():
    # model_1 = SimpleNet(12)
    # model_1.load_state_dict(torch.load('basemodel_full.pt'))
    # fullmodel

    model_1 = BertModel(64, 12, 0.1)
    model_1.load_state_dict(torch.load('fullmodel1.pt'))
    model_2 = SimpleNet(12)
    model_2.load_state_dict(torch.load('basemodel_full.pt'))

    with open("data.pkl",'rb') as fp:
        data = pickle.load(fp)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state = 4)
    test_ds = create_tensors(test_data)

    

    model_1.eval()
    model_2.eval()
    pred_1 = model_1(test_ds['skills'],test_ds['subtests'],test_ds['questions'],test_ds['answers']).squeeze(1)
    # pred_1 = model_1(test_ds['skills']).squeeze(1)

    pred_2 = model_2(test_ds['skills']).squeeze(1)
    # pred_2 = model_2(test_ds['skills'],test_ds['subtests'],test_ds['questions'],test_ds['answers']).squeeze(1)

    b,c = McNemar_test(pred_1,pred_2,test_ds['y'])

    chi_sq = (abs(b-c)-1)**2
    chi_sq = chi_sq/(b+c)

    print('McNemar test statistic:', chi_sq)

    print('p-value:', chi2.sf(chi_sq, 1))


if __name__ == '__main__':
    main()