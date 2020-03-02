#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from config import DefaultConfig
from bert_serving.client import BertClient
from tool import Tools
from gensim.models import word2vec
import random
import time
count=1   
found=1
def cosVector(x,y):
    if(len(x)!=len(y)):
        print('error input,x and y is not in the same space')
        return;
    result1=0.0;
    result2=0.0;
    result3=0.0;
    for i in range(len(x)):
        result1+=x[i]*y[i]   
        result2+=x[i]**2     
        result3+=y[i]**2     
    return result1/((result2*result3)**0.5)
def BugGenerator(w):
    model = word2vec.Word2Vec.load("./text8.model") 
    w_insert=""
    w_delete=""
    w_exchange=""
    w_replace=""
    result=[]
    w1=w2=w3=w4=w5=w
    #insert     
    if len(w1)<=1:
        pass                               
    else:
        w_insert=w1[0:len(w1)//2]+' '+w1[len(w1)//2:]
        result.append(w_insert)
    #delete       
    ran_delete=random.randrange(0,len(w2))
    w_delete=w2.replace(w2[ran_delete],"")
    result.append(w_delete)
    #exchange
    if len(w3)<=1:
        pass            
    else:
        change_index=random.sample(range(0,len(w3)),2)
        w3=list(w3)
        w3[change_index[0]],w3[change_index[1]]=w3[change_index[1]],w3[change_index[0]]
        w_exchange=''.join(w3)     
        result.append(w_exchange)                       
    #replace                                      
    word_replace={'o':'0','l':'1','a':'@','n':'N','k':'K'}
    w4=list(w4)
    for i in range(len(w4)):
        replace_word=word_replace.get(w4[i])
        if replace_word:   
            w4[i]=replace_word        
            break             
    w_replace="".join('%s' %id for id in w4)          
    result.append(w_replace)      
    try:
        word_sim= model.most_similar(w5, topn=5)  # 20个最相关的 
        for item in word_sim:
            result.append(item[0])
        return result
    except:
        pass                              
    return result

def selectBug(w,x,y):
    bugs=BugGenerator(w)        
    bestBug=bugs[0]
    bestBugScore=0
    for bk in bugs:
        x_temp=' '.join(x)                              
        x_temp=x_temp.replace(w,bk)
        score=abs(Fy(' '.join(x),y)-Fy(x_temp,y))              
        if score > bestBugScore:              
            bestBugScore=score
            bestBug=bk                           
    return bestBug
def s(x,x_):      
    bc = BertClient()
    x1=bc.encode([x])
    x2=bc.encode([x_])
    result=cosVector(x1[0],x2[0])
    return result              
                                
def findBestTextBugger(x,gradient_x,y,pos_label):                                                                       
    global count,found
    count =count+1                          
    d={}                              
    x_=x
    k=0.8                                  
    judge=0                                              
    for i in range(len(x)):                     
        d[x[i]]=gradient_x[i]    
    w_order=sorted(d.items(),key=lambda x:x[1],reverse=True)
    total=""    
    for i in range(len(w_order)):         
        if judge==5:
            print("try:",total)       
            print("Not Fount")
            print()           
            return "Not Found!"         
        bug=selectBug(w_order[i][0],x_,y) 
        print(w_order[i][0],"-->",bug)          
        total=total+"type("+ w_order[i][0]+"-->"+bug+"), "                                                   
        x_=' '.join(x_).replace(' '+w_order[i][0]+' ',' '+bug+' ')           
        if s(' '.join(x),x_) <= k:               
            print("Not Found! k<=0.8!")                                     
            return "Not Found! k<=0.8!"              
        if Fl(x_,y):
            c_label=Fy(x_,y)
            origin_label="negative" 
            now_label="negative"          
            if y[0]==1:                       
                origin_label="positive"
            if c_label > 0.5:                   
                now_label="positive"                   
            f=open("./Generate_countermeasure_samples.txt","a")  
                                           
            f.write("No-"+str(found)+"srcSentence:"+' '.join(x)+"\n")               
            f.write("No-"+str(found)+"dstSentence:"+x_+"\n")                                          
            f.write("No-"+str(found)+"label:"+origin_label+"--->"+now_label+"\n")
            f.write("No-"+str(found)+"probability:positive("+str(pos_label)+")"+"-->positive("+str(c_label)+")\n")  
            f.write("No-"+str(found)+"mutate:"+total+"\n")                             
            f.write("\n")                                
                 
            print("No-"+str(found)+"srcSentence:",' '.join(x))                                       
            print("No-"+str(found)+"srcSentence:",x_)
            print("No-"+str(found)+"label:"+origin_label+"--->"+now_label)        
            print("No-"+str(found)+"probability:positive(",pos_label,")","-->positive(",c_label,")")
            print("No-"+str(found)+"mutate:",total)       
            print()                        
            found=found+1            
            return                                    
        x_=(x_.strip().split(" "))             
        judge=judge+1                        
    print("Not Fount") 
    print()        
    return "Not Found!"      
                          
def index_to_word(x):
    wordsList = np.load('wordsList.npy')
    wordsList = wordsList.tolist()
    wordsList = [word.decode('UTF-8') for word in wordsList]
    result=""
    for i in range(len(x)):       
        result +=wordsList[x[i]]+" "          
    return result.strip().split(" ")             
def word_to_index(x):
    wordsList = np.load('wordsList.npy')
    wordsList = wordsList.tolist()
    wordsList = [word.decode('UTF-8') for word in wordsList]
    result=[]
    for i in range(len(x)):
        try:
            result.append(wordsList.index(x[i]))   
        except:
            result.append(399999)                         
    result=result+[0 for i in range(250-len(result))]    
    return result 
def f_fun_comm(x,y):      
    net = SentimentNet(embed_size=opt.embed_size,
               num_hiddens=opt.num_hiddens,
               num_layers=opt.num_layers,         
               bidirectional=opt.bidirectional,           
               weight=opt.wordVectors,
               labels=opt.labels, use_gpu=opt.use_gpu,model_select=opt.model_used)    
    feature=word_to_index(x.split())                     
    net_trained = torch.load('./model_pytorch_gru/model.pkl')
    feature=np.array(feature[0:250]).reshape(1,250)      
    feature=Variable(torch.LongTensor(feature))    
    feature_keep=feature
    feature=net_trained.embedding(feature)                     
    feature = Variable(feature)                 
    label = Variable(torch.LongTensor(y).float()) 
    score,weight = net_trained(feature,feature_keep,label,"test") 
    result=F.softmax(score,dim=1)        
    return result      
def Fy(x,y):       
    result=f_fun_comm(x,y)
    return result.detach().numpy()[0][0]            
def Fl(x,y):       
    result=f_fun_comm(x,y)     
    if result.detach().numpy()[0][0]>0.5:
        chage_label=[1,0]
        return chage_label!=y                            
    else:
        chage_label=[0,1]       
        return chage_label!=y

def get_real_len(x):
    count=0
    for i in range(len(x)-1):
        if x[i]==0 and x[i+1]==0 :
            return count               
        else:        
            count=count+1
    return 250 
def Generate_countermeasure_samples(feature_keepi,label,feature,i,neg_label,weight):                      
    global count                                                                                                                                                           
    real_len=get_real_len(feature_keepi.numpy())    
    gradient_x=[]                                                                                             
    x=feature_keepi[:real_len]  
    weight = weight.detach().numpy()[:real_len]
    attention_top_weight=[(weight[i],x.detach().numpy()[i],i)for i in range(real_len)]    
    attention_top_weight=sorted(attention_top_weight,reverse=True)
    attention_top_word=[one[1] for one in attention_top_weight[:20]] 
    #print(attention_top_weight[:100])        
    x=index_to_word(x.numpy().tolist())        
    #print(index_to_word(attention_top_word))
    for j in range(real_len):                                                                                                                                     
        gradient_x.append(feature.grad[i][j].sum().item())                           
    print(count,".",' '.join(x))       
    findBestTextBugger(x,gradient_x,label,neg_label)

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )
    def forward(self, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights
#############################################模型
class SentimentNet(nn.Module): 
    def __init__(self,  embed_size, num_hiddens, num_layers,
                 bidirectional, weight, labels, use_gpu,model_select, **kwargs):
        super(SentimentNet, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.attention = SelfAttention(num_hiddens)
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(opt.word_embed_size, embed_size)
        self.embedding.weight.data.copy_(torch.from_numpy(opt.wordVectors))
        self.embedding.weight.requires_grad = False   
        if opt.model_used=="gru":
            self.encoder = nn.GRU(input_size=embed_size, hidden_size=self.num_hiddens,
                               num_layers=num_layers, bidirectional=self.bidirectional,batch_first=True)
        elif opt.model_used=="lstm":
            self.encoder = nn.LSTM(input_size=embed_size, hidden_size=self.num_hiddens,
                               num_layers=num_layers, bidirectional=self.bidirectional,batch_first=True)
        if self.bidirectional:
            self.decoder = nn.Linear(num_hiddens * 2, labels)
        else:
            self.decoder = nn.Linear(num_hiddens * 1, labels)
    def get_Squence_from_indexs(self,word_indexs,lengthOfWord):
        word_index={}
        f=open("./word_to_index.txt","r")
        line=f.readline()
        while line:
            word_index[int(line.split()[0])]=line.split()[1]
            line=f.readline()
        result=[]
        for i in range(len(word_indexs)):
            if word_indexs[i]!=0:
                result.append(word_index.get(word_indexs[i])+" ")
        return result
    def forward(self, inputs,input2,label,flag):
        global sequence_count 
     
        opt.batch_size_train=1    

        sen_word_length=[] 
        for x in range(opt.batch_size_train):
            count=0
            for y in range(opt.maxSelength):
                if input2[x][y]!=0:
                    count=count+1      
                else:        
                    break
            sen_word_length.append(count)
        embeddings = inputs     

        out, hidden = self.encoder(embeddings)  
                          
        out_cut=out                    
        for i in range(opt.batch_size_train):
            for j in range(sen_word_length[i]):   
                out_cut[i]=out[i][sen_word_length[i]-1]              
                                                                  
                          
        encoding,attn_weights = self.attention(out_cut) 
        outputs = self.decoder(encoding)  
        return outputs,attn_weights

def train():
    net = SentimentNet(embed_size=opt.embed_size,
                       num_hiddens=opt.num_hiddens,
                       num_layers=opt.num_layers,
                       bidirectional=opt.bidirectional,
                       weight=opt.wordVectors,
                       labels=opt.labels, use_gpu=opt.use_gpu,model_select=opt.model_used)
    loss_function = nn.CrossEntropyLoss( )
    optimizer =optim.Adam(net.parameters(), lr=opt.lr)
    for epoch in range(opt.num_epochs):
        train_loss= 0                      
        train_acc= 0         
        n=0         
        for i,(feature, label) in enumerate(trainloader):  
            if i==len(trainloader)-1:
                break      
            n=n+1                          
            net.zero_grad()      
            feature_keep=feature
            feature=net.embedding(feature)        
            feature = Variable(feature, requires_grad=True)        
            label = Variable(label) 
            score = net(feature,feature_keep,label,"train")  
            if opt.use_gpu:
                feature=feature.cuda()
                label=label.cuda()         
            loss = loss_function(score,torch.max(label, 1)[1])
            loss.backward()    
            optimizer.step()
            acc=accuracy_score(torch.argmax(score.cpu().data,
                                                      dim=1), torch.max(label, 1)[1].cpu())
            train_acc += acc                    
            train_loss += loss
            print('epoch: %d, train loss: %.4f, train acc avg: %.2f ,  acc: %.2f ' %
                  (epoch, train_loss.data / n, (train_acc / n),acc))
        with torch.no_grad():
            m=0                        
            test_acc= 0
            for i,(feature, label) in enumerate(testloader):
                if i==len(trainloader)-1:
                    break
                m=m+1             
                feature_keep=feature
                feature=net.embedding(feature)                    
                feature = Variable(feature)
                label = Variable(label)                                           
                if opt.use_gpu:                                  
                    feature=feature.cuda()                
                    label=label.cuda()
                score = net(feature,feature_keep,label,"test")
                train = accuracy_score(torch.argmax(score.cpu().data,dim=1), torch.max(label, 1)[1].cpu())
                test_acc= test_acc+train
            print('   test acc: %.2f ' %(test_acc/m))
        if opt.model_used=="lstm":                            
            torch.save(net, './model_pytorch_lstm/model.pkl')
        elif opt.model_used=="gru":
            torch.save(net, './model_pytorch_gru/model.pkl')                                 
def test():
    loss_function = nn.CrossEntropyLoss( )
    net = SentimentNet(embed_size=opt.embed_size,  
                       num_hiddens=opt.num_hiddens,
                       num_layers=opt.num_layers,               
                       bidirectional=opt.bidirectional,
                       weight=opt.wordVectors,
                       labels=opt.labels, use_gpu=opt.use_gpu,model_select=opt.model_used)
    if opt.model_used=="lstm":
        net_trained = torch.load('./model_pytorch_lstm/model.pkl')
    elif opt.model_used=="gru":
        net_trained = torch.load('./model_pytorch_gru/model.pkl')
    test_acc=0
    m=1
    for i,(feature, label) in enumerate(testloader):     
        if i==len(trainloader)-1:      
            break                                               
        net_trained.zero_grad()                   
        label = Variable(label)
        feature[0]=torch.Tensor((word_to_index("these immortal lines begin the jack unk directed unk dion brothers the plot centers around two blue collar west virginian brothers stacy keach and frederic forrest who commit robberies in hopes of using the money to open a seafood restaurant what follows is quite an adventure and many comedic events ensue the action scenes are all top notch and consist of some nicely realized shootouts the latter of which is absolutely amazing and occurs in an abandoned building being demolished by a wrecking ball the film was written by now famous director terrence malick and features an early appearance by margot kidder all in all an excellent hidden gem of the 70s and easily one of the finest unk hybrids every made hopefully it gets a decent widescreen dvd release".split())))[0:250]
        feature[1]=torch.Tensor((word_to_index("man if anyone was obliged a great zombie movie after reading that title then you are a retard and you deserve to be disappointed as for myself i was obliged a unk cheeseball zombie flick and thats exactly what i got i wasnt disappointed at all i thought it was a cool little movie the zombies were exactly as they should be because all of the zombies had just been turned so they are unk zombies obviously they did that because it unk been funny expensive if they had done unk rotted zombie fx i understood the whole thing i have no idea how anyone could seriously nitpick this movie its called hood of the living dead for the love of god would you watch redneck zombies and any uwe boll movie and actually expect it to be great of course not so why there are some morons on imdb whining like school girls about this movie ill never understand oh and yes there are worse movies out there so stop saying that this was the greatest unk ever seen cause you know youre full of it you ever watch unk or unk or house of the dead those are some of the greatest ive ever seen if you cant see that its just a low budgeted zombie movie obviously made by zombie movie fans then somethings wrong with you i just had fun with it thumbs up from me and id also like to see a".split())))[0:250]

        feature_keep=feature
        feature=net_trained.embedding(feature)   
        feature = Variable(feature, requires_grad=True)        
        label = Variable(label.float(), requires_grad=True) 
        if opt.use_gpu:                                
            feature=feature.cuda()   
            label=label.cuda()
        score,weight = net_trained(feature,feature_keep,label,"test")          
        forecast_labels=F.softmax(score,dim=1).detach().numpy()   
        gradients=torch.ones(48,2)  
        score.backward(gradient=gradients)           
        for j in range(48):                                                                                       
            if forecast_labels[j][0]>0.5:                                                        
                forecast_label=[1,0]                               
            else:                                                  
                forecast_label=[0,1]          
 
            Generate_countermeasure_samples(feature_keep[j],forecast_label,feature,j,forecast_labels[j][0],weight[j])
        train = accuracy_score(torch.argmax(score.cpu().data,dim=1), torch.max(label, 1)[1].cpu())
        test_acc= test_acc+train
        m=m+1
    print('test acc: %.2f ' %(test_acc/m))
                               
if __name__ == "__main__":            
    opt = DefaultConfig()                            
    tools = Tools()     
    inputs_,lables_=tools.get_data(opt) 
    trainloader,testloader=tools.split_data(opt,inputs_,lables_) 
    #train()                             
    test()               
