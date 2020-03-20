# Adversarial_Examples_Generate 

------

Description:  A white box algorithm that generate adversarial examples according to the paper  
《TEXTBUGGER: Generating Adversarial Text AgainstReal-world Applications》

## Pre knowledge
> * this model is about Sentiment analysis which dataset is IMDB  
> * i provided a  model in folder named model_pytorch_gru to you for easy begin
## Requirement
> * python3
> * pytorch
> * numpy==1.16.0
> * torch==1.3.1
> * gensim==3.8.1
> * bert_serving_client==1.9.9
> * bert_serving==0.0.1
> * scikit_learn==0.22.2

## Easy Begin
> just run:python pytorch.py  
then you will see the result file Generate_countermeasure_samples.txt

## A example about result file
>**No-3srcSentence**:the protector you hear the name you think ah its a crappy hong kong movie guess what its not hong kong and yes it is crappy this amazingly **stupid** jackie chan film ruined by us yes us the americans im boiling with anger ooh i think ill jump out that window has chan as a new york cop hunting down a gang avenging the death of his buddy sounds unk its not dont waste your money renting it to prove he could make a better cop film chan made the amazing police story   
**No-3srcSentence**:the protector you hear the name you think ah its a crappy hong kong movie guess what its not hong kong and yes it is crappy this amazingly **funny** jackie chan film ruined by us yes us the americans im boiling with anger ooh i think ill jump out that window has chan as a new york cop hunting down a gang avenging the death of his buddy sounds unk its not dont waste your money renting it to prove he could make a better cop film chan made the amazing police story  
**No-3label**:negative--->positive  
**No-3probability**:positive(0.05022594)-->positive(0.815222)       
**No-3mutate**:type(stupid-->funny)  

**explanation**  
第一行是原始的句子  
第二行是产生的对抗样本  
第三行分别对应原始标签和对抗样本的标签  
第四行是概率的变化，我们可以看到原始的句子判断为positive的概率0.05，对抗样本判断为positive的概率是0.81   
第五行展示了生成这个对抗样本具体改动了哪些词,比如这里就是将原始句子的stupid改为了funny

# 核心原理简要说明
>本算法根据梯度选出一个句子中topN个重要的词，按词的重要性依次做下列几个操作  
1.对该词随机加入空格  
2.对该词随机删除个别字母  
3.替换该词为其近视词  
4.相似字母替换，如a->@ l->1....  
...  
一直做上述操作直到原始句子的label发生改变  
  
还有一些数据文件太大了，放在云盘链接: https://pan.baidu.com/s/1d00LxLgWhqNoPUJLlK4-CQ 提取码: ba7j 
