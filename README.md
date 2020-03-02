# Adversarial_Examples_Generation 

------

description:  A white box algorithm that generate adversarial examples according to the paper 《TEXTBUGGER: Generating Adversarial Text AgainstReal-world Applications》

##Pre knowledge
1.this model is about Sentiment analysis which data set is IMDB 
2.i provided a pre training model in folder named model_pytorch_gru
##Requirement
> * python3
> * pytorch
> * numpy==1.16.0
> * torch==1.3.1
> * gensim==3.8.1
> * bert_serving_client==1.9.9
> * bert_serving==0.0.1
> * scikit_learn==0.22.2

##Easy Begin
> just run:python pytorch.py 
then you will see the result file Generate_countermeasure_samples.txt

## Some examples about result file
>$\color{red}{No-3srcSentence:}$the protector you hear the name you think ah its a crappy hong kong movie guess what its not hong kong and yes it is crappy this amazingly $\color{red}{stupid}$ jackie chan film ruined by us yes us the americans im boiling with anger ooh i think ill jump out that window has chan as a new york cop hunting down a gang avenging the death of his buddy sounds unk its not dont waste your money renting it to prove he could make a better cop film chan made the amazing police story 
$\color{red}{No-3srcSentence:}$the protector you hear the name you think ah its a crappy hong kong movie guess what its not hong kong and yes it is crappy this amazingly $\color{red}{funny}$ jackie chan film ruined by us yes us the americans im boiling with anger ooh i think ill jump out that window has chan as a new york cop hunting down a gang avenging the death of his buddy sounds unk its not dont waste your money renting it to prove he could make a better cop film chan made the amazing police story
$\color{red}{No-3label:}$negative--->positive
$\color{red}{No-3probability:}$positive(0.05022594)-->positive(0.815222)    
$\color{red}{No-3mutate:}$type(stupid-->funny)

$\color{red}{explanation}$ ：
第一行是原始的句子
第二行是产生的对抗样本
第三行分别对应原始标签和对抗样本的标签
第四行是概率的变化，我们可以看到原始的句子判断为positive的概率0.05022594，对抗样本判断为positive的概率是0.815222
第五行展示了生成这个对抗样本具体改动了哪些词
