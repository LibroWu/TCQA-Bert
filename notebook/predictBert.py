import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer, BertModel
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import csv
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from PIL import Image, ImageFile
import torchvision.transforms as transforms
from torchvision import models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch.optim as optim
import os
import json
from transformers import BertTokenizer, BertModel
import operator
class ContextQuType:
    def __init__(self, tokenizer, length, ids, context, question):
        self.tokenizer = tokenizer
        self.codeLength = length
        self.ids = ids
        self.context = context
        self.question = question
        self.after_encode = 0
        
    def convert(self):
        if self.after_encode==0:
            after_encode = tokenizer(self.question,self.context,max_length=self.codeLength,padding="max_length")
        return torch.stack([torch.tensor(after_encode['input_ids'][:512]),torch.tensor(after_encode['token_type_ids'][:512]),torch.tensor(after_encode['attention_mask'][:512])])
        
class LabelType:
    def __init__(self, tokenizer, length, CQu, has_ans, text, start):
        self.tokenizer = tokenizer
        self.codeLength = length
        self.has_ans = has_ans
        self.text = text
        self.start = start
        self.CQu = CQu
    
    def convert(self):
        # make groundtruth
        S = [0.0] * self.codeLength
        E = [0.0] * self.codeLength
        background = tokenizer.encode(self.CQu.question,self.CQu.context,max_length=self.codeLength,padding="max_length")
        background = background[:512]
        if self.has_ans:
            after_encode = tokenizer.encode(self.text)[1:-1]
            ans_length = len(after_encode)
            for idx in range(len(background)):
                if operator.eq(background[idx:idx+ans_length],after_encode):
                    S[idx] = 1.0
                    E[idx+ans_length-1] = 1.0
                    break
        else:
            S[0] = 1.0
            E[0] = 1.0
        return torch.stack([torch.tensor(S),torch.tensor(E)])
class OutputPair:
    def __init__(self,logits_start,logits_end):
        self.logits_start = logits_start
        self.logits_end = logits_end

class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()
    def forward(self,outputs,labels):
        log_soft = F.log_softmax(outputs.logits_start,dim=1)
        loss_start = F.cross_entropy(log_soft, labels[:,0])
        loss_end   = F.cross_entropy(F.log_softmax(outputs.logits_end,dim=1), labels[:,1])
        return loss_start + loss_end / 2.0

class OutputPair:
    def __init__(self,logits_start,logits_end):
        self.logits_start = logits_start
        self.logits_end = logits_end

class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()
    def forward(self,outputs,labels):
        log_soft = F.log_softmax(outputs.logits_start,dim=1)
        loss_start = F.cross_entropy(log_soft, labels[:,0])
        loss_end   = F.cross_entropy(F.log_softmax(outputs.logits_end,dim=1), labels[:,1])
        return loss_start + loss_end / 2.0

class BERTQuA(nn.Module):
    def __init__(self, bert, tokenizer,code_length,device,epsilon = 1.0):
        super(BERTQuA, self).__init__()
        self.tokenizer = tokenizer
        self.device = device
        self.encoder = bert.to(device)
        self.code_length = code_length
        self.output_start = nn.Linear(768,1).to(device)
        self.output_end = nn.Linear(768,1).to(device)
        self.epsilon = 1.0

    def forward(self, inputs):
        tokens_X, segments_X, masks = inputs[:,0],inputs[:,1],inputs[:,2]
        encoded_X = self.encoder(tokens_X, token_type_ids=segments_X, attention_mask=masks).last_hidden_state
        return OutputPair(self.output_start(encoded_X),self.output_end(encoded_X))
        
    def prediction(self,question,answer_text):
        ctx = ContextQuType(self.tokenizer,self.code_length,'',answer_text,question)
        inputs = torch.stack([ctx.convert()]).to(device)
        token_ids = tokenizer(question,answer_text,max_length=self.code_length,padding="max_length")['input_ids']
        outputs = self.forward(inputs)
        score_len = outputs.logits_start.shape[1]
        start_scores = outputs.logits_start.reshape(score_len)
        end_scores = outputs.logits_end.reshape(score_len)
        no_ans_scores = start_scores[0] + end_scores[0]
        answer = ''
        flag = False
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        if (torch.max(start_scores) + torch.max(end_scores) <= no_ans_scores +  self.epsilon):
            answer = ''
        else:
            answer_start = torch.argmax(start_scores)
            answer_end = torch.argmax(end_scores)
            for i in range(answer_start,answer_end+1):
                if "##" in tokens[i]:
                    answer+=tokens[i][2:]
                else:
                    if tokens[i]!="." and tokens[i]!=',' and flag:
                        answer+=' '
                    answer+=tokens[i]
                    if tokens[i]!=",":
                          flag = True
        return answer
    def load_data(self,file_path):
        CQu, La = load_data(file_path,self.tokenizer,self.code_length)
        self.inputs = torch.stack([ctx.convert() for ctx in CQu])
        self.labels = torch.stack([ctx.convert() for ctx in La])
    
    def train(self, epochs=3, batch_size = 48, lr = 3e-5, T=10):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = CrossEntropy().to(device)
        self.train_loader = DataLoader(TensorDataset(self.inputs,self.labels),batch_size = batch_size, shuffle = True, num_workers = 2)
        epoch_loss = []
        for epoch in range(epochs):      
            running_loss = 0.0
            epoch_running_loss = 0.0
            batch_count = 0
            for batchidx, (x, label) in enumerate(self.train_loader):
                x, label = x.to(device), label.to(device)
                output = self.forward(x)
                output.logits_start=output.logits_start.resize(len(output.logits_start),self.code_length)
                output.logits_end=output.logits_end.resize(len(output.logits_end),self.code_length)
                loss = criterion(output, label)
                # backprop
                optimizer.zero_grad()  #?????????0
                loss.backward()   #????????????
                optimizer.step()   #????????????
                
                running_loss += loss.item()
                epoch_running_loss += loss.item()
                batch_count += 1
                if batchidx % T == T-1:
                    print(epoch,' batchidx: ', batchidx, ' loss: ', running_loss/T)
                    running_loss = 0.0
            epoch_loss.append(epoch_running_loss/batch_count)
            print(epoch, 'loss:', epoch_running_loss/batch_count)
        return epoch_loss


# model = BertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')
# model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
# model = BertForQuestionAnswering.from_pretrained('deepset/bert-large-uncased-whole-word-masking-squad2')
checkpoints = torch.load('TryEpochs4.pkl')  #????????????????????????????????????
checkpoint = checkpoints['state_dict']
step = checkpoints['epoch']   #???????????????
device = torch.device('cuda')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
pre_trained_model = BertModel.from_pretrained("bert-base-uncased")
model = BERTQuA(pre_trained_model,tokenizer,512,device)
model.load_state_dict(checkpoint)

#tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
#tokenizer = BertTokenizer.from_pretrained('deepset/bert-large-uncased-whole-word-masking-squad2')
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
# model = DistilBertModel.from_pretrained("distilbert-base-uncased-distilled-squad")
import json

with open("dev-v2.0.json") as json_data:
  dev = json.load(json_data)['data']

res = {}

for data in dev:
  for paragraphs in data['paragraphs']:
    context = paragraphs['context']
    for qas in paragraphs['qas']:
      question = qas['question']
      res[qas['id']] = model.prediction(question,context)

with open("prediction.json","w") as os:
  json.dump(res,os)
