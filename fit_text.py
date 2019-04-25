import pandas as pd
import numpy as np
import re
import jieba
import codecs
from collections import Counter

np.random.seed(50)
data_dir_path = './data'
model_dir_path = "./model"
MAX_INPUT_WORD_SIZE = 1000
MAX_TARGET_WORD_SIZE = 100

MAX_INPUT_VOCAB_SIZE = 21000
MAX_TARGET_VOCAB_SIZE = 3500


def text_cut():
    f1 = codecs.open("./data/cut_title.txt", 'w', 'utf-8')
    f2 = codecs.open("./data/cut_context.txt", 'w', 'utf-8')
    print("loading data . . .")
    file = open(data_dir_path + '/data.txt')
    file_string = ""
    for i in file:
        file_string += i
    document = re.compile('<doc>' + '(.*?)' + '</doc>', re.S)
    title = re.compile('<contenttitle>' + '(.*?)' + '</contenttitle>', re.S)
    context = re.compile('<content>' + '(.*?)' + '</content>', re.S)
    documents = document.findall(file_string)
    titles, contexts = [], []
    for lines in documents:
        titles_tmp = title.findall(lines)
        contexts_tmp = context.findall(lines)
        contexts_tmp = str(contexts_tmp).replace('\\n', '')
        titles.append(titles_tmp)
        contexts.append(contexts_tmp)

    X, Y = [], []
    for line in titles:
        # 结巴分词的结果只能使用一次，使用.join()后该对象变为空
        line = str(line).replace("['", "").replace("']", "").replace("'， '", "")
        title_cut = jieba.cut(line, cut_all=False)
        f1.write(" ".join(title_cut))
        f1.write("\n")
        # 结巴分词的结果只能使用一次，使用.join()后该对象变为空
        tmp = jieba.cut(line, cut_all=False)
        str1=""
        for i in tmp:
            str1+=str(i)
            str1+=" "
        X.append(str1)
    for line in contexts:
        line = str(line).replace("['", "").replace("']", "").replace("'， '", "")
        context_cut = jieba.cut(line, cut_all=False)
        f2.write(" ".join(context_cut))
        f2.write("\n")
        # 结巴分词的结果只能使用一次，使用.join()后该对象变为空
        tmp = jieba.cut(line, cut_all=False)
        str2=""
        for i in tmp:
            str2+=str(i)
            str2+=" "
        Y.append(str2)
    f1.close()
    f2.close()
    # print(len(X))
    # print(len(Y))
    return X,Y
    


def cipin():
    fx2 = open("./data/train_1/cut_title.txt", encoding='utf-8')
    fx1 = open("./data/train_1/cut_context.txt", encoding='utf-8')
    x, y = [], []
    for line in fx1:
        x.append(line)
    for line in fx2:
        y.append(line)
    input_counter = Counter()
    target_counter = Counter()
    max_input_seq_length = 0
    max_target_seq_length = 0
    for line in x:
        line = str(line).replace("\n", '').replace('图文', '').split(' ')
        line = [x for x in line if x != "" and x != "]"]
        line = [x for x in line if x != ":" and x != "-" and x != "["]
        line = [x for x in line if x != "," and x != "(" and x != ")"]
        seq_length = len(line)
        if len(line) > MAX_INPUT_WORD_SIZE:
            line = line[0:MAX_INPUT_WORD_SIZE]
            seq_length = len(line)
        for word in line:
            input_counter[word] += 1
        max_input_seq_length = max(max_input_seq_length, seq_length)
    
    for line in y:
        line = 'START ' + str(line) + ' END'
        line = line.replace("\n", '').replace('图文', '').split(' ')
        line = [x for x in line if x != "" and x != "]"]
        line = [x for x in line if x != ":" and x != "-" and x != "["]
        line = [x for x in line if x != "," and x != "(" and x != ")"]
        seq_length = len(line)
        if len(line) > MAX_TARGET_WORD_SIZE:
            line = line[0:MAX_TARGET_WORD_SIZE]
            seq_length = len(line)
        for word in line:
            target_counter[word] += 1
        # print(line)
        max_target_seq_length = max(max_target_seq_length, seq_length)
    
    # print(len(input_counter))
    # print(len(target_counter))
    
    input_word2idx = dict()
    for idx, word in enumerate(input_counter.most_common(MAX_INPUT_VOCAB_SIZE)):
        input_word2idx[word[0]] = idx + 2
    input_word2idx['PAD'] = 0
    input_word2idx['UNK'] = 1
    input_idx2word = dict([(idx, word) for word, idx in input_word2idx.items()])
    # print(len(input_idx2word))
    
    target_word2idx = dict()
    for idx, word in enumerate(target_counter.most_common(MAX_TARGET_VOCAB_SIZE)):
        target_word2idx[word[0]] = idx + 1
    target_word2idx['UNK'] = 0
    target_idx2word = dict([(idx, word) for word, idx in target_word2idx.items()])
    # print(len(target_counter))
    
    num_input_tokens = len(input_word2idx)
    num_target_tokens = len(target_word2idx)
    
    config = dict()
    config['input_word2idx'] = input_word2idx
    config['input_idx2word'] = input_idx2word
    config['target_word2idx'] = target_word2idx
    config['target_idx2word'] = target_idx2word
    config['num_input_tokens'] = num_input_tokens
    config['num_target_tokens'] = num_target_tokens
    config['max_input_seq_length'] = max_input_seq_length
    config['max_target_seq_length'] = max_target_seq_length
    
    # print("num_input_tokens:",config["num_input_tokens"])
    # print("num_target_tokens:",config["num_target_tokens"])
    # print("max_input_seq_length:",config["max_input_seq_length"])
    # print("max_target_seq_length:",config["max_target_seq_length"])
    return x,y,config
# cipin()
    
def read_txt():
    # savefile = "./data/new_news.txt"
    # f1 = open(savefile,"w", encoding="utf-8")
    #
    #
    # file="./data/news_tensite_xml.dat"
    # f  = open(file,encoding="gbk",errors="ignore")
    #
    # news = []
    # for i in f:
    #     news.append(str(i))
    #     # print(str(i))
    # for i in news:
    #     f1.write(i)
    # f.close()
    # f1.close()
    
    
    file_path = "./data/new_news.txt"
    f1 = open(file_path,"r", encoding="utf-8")
    text = []
    # print("begin")
    for i in f1:
        text.append(i)
        # print(i)
    context_file_path = "./data/train/document.txt"
    f2 = open(context_file_path,"w",encoding="utf-8")

    title_path = "./data/train/titles.txt"
    f_title = open(title_path,"w",encoding="utf-8")

    context_path = "./data/train/context.txt"
    f_context = open(context_path,"w",encoding="utf-8")

    i, tmp0 = 0, []
    title = re.compile('<contenttitle>' + '(.*?)' + '</contenttitle>', re.S)
    context = re.compile('<content>' + '(.*?)' + '</content>', re.S)
    count = 1
    while i < len(text)-6:
        string1 = "".join(text[i:i+6])
        string1.replace("\n","")
        if title.findall(string1)[0] and context.findall(string1)[0]:
            f_title.write(title.findall(string1)[0]+"\n")
            f_context.write(context.findall(string1)[0]+"\n")
            # print("success",count)
            count+=1
            if count%5000==0:
                break
        i+=6
        # f2.write(string1)
    # print(i)
    f1.close()
    f2.close()
    
    # context_file_path = "./data/train/document.txt"
    # f2 = open(context_file_path,"r",encoding="utf-8")
    # x=[]
    # title = re.compile('<url>' + '(.*?)' + '</url>', re.S)
    # a=0
    # for i in f2:
    #     i = i.replace("\n","")
    #     print(i,a)
    #     # print(title.findall(i),"aaa")
    #     # print(a)
    #     a+=1
    #
    # title_path = "./data/train/titles.txt"
    # f_title = open(title_path,"r",encoding="utf-8")
    # for i in f_title:
    #     print(i)
    # pass

# read_txt()

def new_cipin():
    fx1 = open("./data/train/context.txt", encoding='utf-8')
    fx2 = open("./data/train/titles.txt", encoding='utf-8')
    x, y = [], []
    for line in fx1:
        x.append(line)
    for line in fx2:
        y.append(line)
    # print(len(x))
    # print(len(y))
    input_counter = Counter()
    target_counter = Counter()
    max_input_seq_length = 0
    max_target_seq_length = 0
    for line in x:
        line = str(line).replace("\n", '').replace('图文', '').split(' ')
        line = [x for x in line if x != "" and x != "]"]
        line = [x for x in line if x != ":" and x != "-" and x != "["]
        line = [x for x in line if x != "," and x != "(" and x != ")"]
        seq_length = len(line)
        if len(line) > MAX_INPUT_WORD_SIZE:
            line = line[0:MAX_INPUT_WORD_SIZE]
            seq_length = len(line)
        for word in line:
            input_counter[word] += 1
        max_input_seq_length = max(max_input_seq_length, seq_length)
    
    for line in y:
        line = 'START ' + str(line) + ' END'
        line = line.replace("\n", '').replace('图文', '').split(' ')
        line = [x for x in line if x != "" and x != "]"]
        line = [x for x in line if x != ":" and x != "-" and x != "["]
        line = [x for x in line if x != "," and x != "(" and x != ")"]
        seq_length = len(line)
        if len(line) > MAX_TARGET_WORD_SIZE:
            line = line[0:MAX_TARGET_WORD_SIZE]
            seq_length = len(line)
        for word in line:
            target_counter[word] += 1
        max_target_seq_length = max(max_target_seq_length, seq_length)
    
    # print(len(input_counter))
    # print(len(target_counter))
    # print(input_counter)
    # print(target_counter)
    
    input_word2idx = dict()
    for idx, word in enumerate(input_counter.most_common(MAX_INPUT_VOCAB_SIZE)):
        input_word2idx[word[0]] = idx + 2
    input_word2idx['PAD'] = 0
    input_word2idx['UNK'] = 1
    input_idx2word = dict([(idx, word) for word, idx in input_word2idx.items()])
    # print(len(input_idx2word))

    target_word2idx = dict()
    for idx, word in enumerate(target_counter.most_common(MAX_TARGET_VOCAB_SIZE)):
        target_word2idx[word[0]] = idx + 1
    target_word2idx['UNK'] = 0
    target_idx2word = dict([(idx, word) for word, idx in target_word2idx.items()])
    # print(len(target_counter))

    num_input_tokens = len(input_word2idx)
    num_target_tokens = len(target_word2idx)

    config = dict()
    config['input_word2idx'] = input_word2idx
    config['input_idx2word'] = input_idx2word
    config['target_word2idx'] = target_word2idx
    config['target_idx2word'] = target_idx2word
    config['num_input_tokens'] = num_input_tokens
    config['num_target_tokens'] = num_target_tokens
    config['max_input_seq_length'] = max_input_seq_length
    config['max_target_seq_length'] = max_target_seq_length
    
    # print("num_input_tokens:",config["num_input_tokens"])
    # print("num_target_tokens:",config["num_target_tokens"])
    # print("max_input_seq_length:",config["max_input_seq_length"])
    # print("max_target_seq_length:",config["max_target_seq_length"])
    return x,y,config

# new_cipin()