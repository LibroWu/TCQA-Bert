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
from transformers import XLNetModel, XLNetTokenizerFast
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
        return torch.stack([torch.tensor(after_encode['input_ids'][-512:]),torch.tensor(after_encode['token_type_ids'][-512:]),torch.tensor(after_encode['attention_mask'][-512:])])
        
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
        length_exceed = False
        if len(background) > self.codeLength:
            length_exceed = True
        if self.has_ans:
            after_encode = tokenizer.encode(self.text)[:-2]
            ans_length = len(after_encode)
            after_answer = tokenizer.encode(self.CQu.context[self.start:])
            if length_exceed:
                start_id = len(background) - len(after_answer)
                end_id = start_id + ans_length -1
                if tokenizer.decode(background[start_id:end_id+1])!=self.text:
                    lower = max(start_id-3,0)
                    upper = min(end_id+4,len(background))
                    ans_found = False
                    for i in range(lower,upper):
                        for j in range(i+1,upper):
                            if tokenizer.decode(background[i:j])==self.text:
                                start_id = i
                                end_id = j-1
                                ans_found = True
                                break
                            if ans_found:
                                break
                if end_id >= self.codeLength:
                    end_id = self.codeLength-1
                if start_id >= self.codeLength:
                    start_id = 0
                    end_id = 0
                S[start_id] = 1.0
                E[end_id] = 1.0
            else:
                start_id = len(background) - len(after_answer)
                end_id = start_id + ans_length -1
                if tokenizer.decode(background[start_id:end_id+1])!=self.text:
                    lower = max(start_id-3,0)
                    upper = min(end_id+4,len(background))
                    ans_found = False
                    for i in range(lower,upper):
                        for j in range(i+1,upper):
                            candidate = tokenizer.decode(background[i:j])
                            if len(candidate)>0:
                                if candidate[0]=='$' and self.text[0]!='$':
                                    candidate=candidate[1:]
                                if candidate[0]!='$' and self.text[0]=='$':
                                    candidate='$'+candidate
                            if candidate==self.text:
                                start_id = i
                                end_id = j-1
                                ans_found = True
                                break
                            if ans_found:
                                break
                S[start_id] = 1.0
                E[end_id] = 1.0
            #if tokenizer.decode(background[start_id:end_id+1])!=self.text:
                #print(tokenizer.convert_ids_to_tokens(background))
                #print(tokenizer.decode(after_answer[:ans_length])+'|||'+tokenizer.decode(background[start_id:end_id+1])+'|||'+self.text)
                #print(len(background),len(after_answer),start_id,end_id)
                #print(self.CQu.context[self.start:self.start+len(self.text)]+'|||'+tokenizer.decode(after_encode)+'|||',tokenizer.convert_ids_to_tokens(after_encode))
        else:
            S[-1] = 1.0
            E[-1] = 1.0
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
        start_scores = outputs.logits_start.reshape(-1)
        end_scores = outputs.logits_end.reshape(-1)
        no_ans_scores = start_scores[-1] + end_scores[-1]
        answer = ''
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        #print(outputs.logits_start.reshape(-1))
        #print(outputs.logits_end.reshape(-1))
        if (torch.max(start_scores) + torch.max(end_scores) <= no_ans_scores +  self.epsilon):
            answer = ''
        else:
            add_space = False
            answer_start = torch.argmax(start_scores)
            answer_end = torch.argmax(end_scores)
            answer = self.tokenizer.decode(token_ids[answer_start:answer_end+1])
        return answer
    def predictionLabel(self,label):
        #ctx = ContextQuType(self.tokenizer,self.code_length,'',answer_text,question)
        #inputs = torch.stack([ctx.convert()]).to(device)
        #token_ids = tokenizer(question,answer_text,max_length=self.code_length,padding="max_length")['input_ids']
        #outputs = self.forward(inputs)
        outputs = label.convert()
        #print(outputs)
        token_ids = tokenizer(label.CQu.question,label.CQu.context,max_length=self.code_length,padding="max_length")['input_ids']
        start_scores = outputs[0,:]
        end_scores = outputs[1,:]
        #print(start_scores,end_scores)
        no_ans_scores = start_scores[-1] + end_scores[-1]
        answer = ''
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        #print(outputs.logits_start.reshape(-1))
        #print(outputs.logits_end.reshape(-1))
        if (torch.max(start_scores) + torch.max(end_scores) <= no_ans_scores +  self.epsilon):
            answer = ''
        else:
            answer_start = torch.argmax(start_scores)
            answer_end = torch.argmax(end_scores)
            answer = self.tokenizer.decode(token_ids[answer_start:answer_end+1])
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
                optimizer.zero_grad()  #梯度清0
                loss.backward()   #梯度反传
                optimizer.step()   #保留梯度

                running_loss += loss.item()
                epoch_running_loss += loss.item()
                batch_count += 1
                if batchidx % T == T-1:
                    print(epoch,' batchidx: ', batchidx, ' loss: ', running_loss/T)
                    running_loss = 0.0
            epoch_loss.append(epoch_running_loss/batch_count)
            print(epoch, 'loss:', epoch_running_loss/batch_count)
        return epoch_loss
    def printLabel(self,file_path):
        CQu, La = load_data(file_path,self.tokenizer,self.code_length)
        count = 0
        precise = 0
        for label in La:
            count += 1
            predict = self.predictionLabel(label)
            if predict==label.text:
                precise += 1
            else:
                print(predict,label.text,label.has_ans)
        print(precise/count)

# model = BertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')
# model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
# model = BertForQuestionAnswering.from_pretrained('deepset/bert-large-uncased-whole-word-masking-squad2')
checkpoints = torch.load('xlnetCluster23.pkl')  #是字典型，包含训练次数等
#checkpoints = torch.load('classifySimBertCluster2-6.pkl')  #是字典型，包含训练次数等
checkpoint = checkpoints['state_dict']
step = checkpoints['epoch']   #训练的批次
device = torch.device('cuda')
tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")
pre_trained_model = XLNetModel.from_pretrained("xlnet-base-cased")
model = BERTQuA(pre_trained_model,tokenizer,512,device)
model.load_state_dict(checkpoint)

#tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
#tokenizer = BertTokenizer.from_pretrained('deepset/bert-large-uncased-whole-word-masking-squad2')
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
# model = DistilBertModel.from_pretrained("distilbert-base-uncased-distilled-squad")
import json

with open("dev-v2.0-Tag1.0.json") as json_data:
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
