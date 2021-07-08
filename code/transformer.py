import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, models
from torch.nn.modules.linear import Linear
import transformers

class Transformer(nn.Module):
    # https://galhever.medium.com/sentiment-analysis-with-pytorch-part-4-lstm-bilstm-model-84447f6c4525
    def __init__(self, sentence_dim, skill_dim, dropout):
        super().__init__()
        self.encoder = SentenceTransformer('paraphrase-mpnet-base-v2')
        self.fc1 = nn.Linear(768,sentence_dim)
        self.fc2 = nn.Linear(skill_dim,skill_dim*2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(sentence_dim+skill_dim*2,1)
        self.act = nn.Sigmoid()
    
    def forward(self, text, skills):
        # text = [batch size,sent_length]
        embedded = self.encoder.encode(text)
        x1 = self.fc1(embedded)
        x2 = self.fc2(skills)
        x = torch.cat((x1,x2),dim=1)
        x = self.fc3(self.relu(x))
        pred = self.act(x)
        
        return pred
    

def main():
    
    

if __name__ == '__main__':
    main()