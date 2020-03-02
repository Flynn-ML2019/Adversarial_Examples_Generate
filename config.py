# coding:utf8    
import numpy as np
class DefaultConfig(object):
    ids = np.load('./all.npy',allow_pickle=True)#加载句子 5w条句子各一半
    wordsList = np.load('wordsList.npy')
    wordsList = wordsList.tolist()
    wordsList = [word.decode('UTF-8') for word in wordsList]
    wordVectors = np.load('./wordVectors.npy')#加载训练好的词向量 40w词 一个词50维
    num_layers =3 #隐藏层层数
    num_hiddens=10 #hidden的数量
    model_used="gru"
    bidirectional = False #是否采用双向lstm
    lr =0.01 #学习速率       
    use_gpu = False     
    labels = 2 #标签个数                        
    num_epochs = 5 #迭代次数  一次迭代所有训练数据             
    batch_size_train=48 #喂入数据的batch_size
    embed_size = 50 #每个单词嵌入的维度
    maxSelength=250 #一句话的最大长度
    allSequence=50000 #总共的句子数
    word_embed_size=400000 #训练好的40w的词嵌入
    sequence_count=1
