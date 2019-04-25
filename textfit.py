#conding=utf-8
import re
import jieba
from collections import Counter

MAX_INPUT_SEQ_LENGTH = 500     #每篇文章的单词数量
MAX_TARGET_SEQ_LENGTH = 50      #每篇文章标题的单词数量
MAX_INPUT_VOCAB_SIZE = 8000       #训练集所有文章的单词总数量
MAX_TARGET_VOCAB_SIZE = 3000         #训练集所有文章标题的单词总数量

def DBC2SBC(ustring):
    # ”' 全角转半角 ”'
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
        if not (0x0021 <= inside_code and inside_code <= 0x7e):
            rstring += uchar
            continue
        rstring += chr(inside_code)
    return rstring


def read_text(path):
    """"
    返回新闻的标题和正文信息
    a:title
    b:context
    """
    f = open(path,'r',encoding="gbk",errors="ignore")
    text, co="",0
    for line in f:
        text+= line
        co+=1
        if co>1200:
            break
    newtext=DBC2SBC(text).split("<doc>")
    # print(newtext)
    a,b = [], []
    for item in newtext:
        re1 = re.compile(r'<contenttitle>(.*?)</contenttitle>')
        title=re1.findall(item)

        re2 = re.compile(r'<content>(.*?)</content>')
        content = re2.findall(item)

        if len(title) > 0 and len(content) > 0:
            title1 = title[0].replace("\u3000", " ")
            a.append(title1)
            content = content[0].replace("\u3000", " ")
            b.append(content)
    return a,b

def stop_word(word_list,stop_path):
    """"
    去停用词，停用词在stop_path.txt中
    """
    file = open(stop_path,'r')
    text = file.read().split("\n")
    re_text = []
    for item in word_list:
        if item not in text:
            re_text.append(item)
    return re_text

def data_preprocess(word_list):
    """"
    替换时间，日期，英文等
    """
    new_word =[]
    fuhao=['「','」','￥','…','”','“','》','《','*','\ue40c',',','/','-','[详细]','、']   #特殊符号
    for item in word_list:
        if item in fuhao:
            continue
        re1 = re.findall('[a-zA-Z]+', item)    #去除英文
        re2 = re.findall('(\d+)',item)          #去除数字
        re3 = re.findall('[((.*?))]',item)       #去除小括号里面的内容
        re4 = re.findall('[【(.*?)】]',item)       #去除【】里面的内容
        if len(re1)>0:
            new_word.append('TAG_NAME_EN')
        elif len(re2)>0:
            new_word.append('TAG_NUMBER')
            continue
        elif len(re3)>0 or len(re4)>0:
            continue
        elif item in ["年","月","日","时","分"]:
            new_word.append("TAG_DATE")
            continue
        else:
            new_word.append(item)
    return new_word


def fit_text(X, Y, input_seq_max_length=None, target_seq_max_length=None):
    if input_seq_max_length is None:
        input_seq_max_length = MAX_INPUT_SEQ_LENGTH
    if target_seq_max_length is None:
        target_seq_max_length = MAX_TARGET_SEQ_LENGTH
    input_counter = Counter()   #计数器，统计文章中每个单词的数量
    target_counter = Counter()   #计数器，统计文章中每个单词的数量
    max_input_seq_length = 0
    max_target_seq_length = 0

    for line in X:
        # text = [word.lower() for word in line.split(' ')]    #对每篇文章分词，得到词语数组
        seq_length = len(line)                    #文章单词的数量
        if seq_length > input_seq_max_length:     #若该篇文章的单词数量超过设定的阈值
            line = line[0:input_seq_max_length]   #则取前input_seq_max_length个单词
            seq_length = len(line)
        for word in line:                   #统计文章中每个单词出现的次数
            input_counter[word] += 1
        max_input_seq_length = max(max_input_seq_length, seq_length)      #man_inout_length不能超过设定的阈值

    for line in Y:  # 和上面同样的过程处理标题
        # line2 = 'START ' + line.lower() + ' END'
        new1 = ['START']+line+['END']
        # text = [word for word in line2.split(' ')]
        seq_length = len(new1)
        if seq_length > target_seq_max_length:
            line2 = new1[0:target_seq_max_length]
            seq_length = len(new1)
        for word in new1:
            target_counter[word] += 1
            max_target_seq_length = max(max_target_seq_length, seq_length)


    input_word2idx = dict()
    for idx, word in enumerate(input_counter.most_common(MAX_INPUT_VOCAB_SIZE)):  #返回数量最多的前 MAX_INPUT_VOCAB_SIZE=5000 个元素
        input_word2idx[word[0]] = idx + 2
    input_word2idx['PAD'] = 0
    input_word2idx['UNK'] = 1
    input_idx2word = dict([(idx, word) for word, idx in input_word2idx.items()])

    # print(input_idx2word)   #{0: 'PAD', 1: 'UNK', 2: 'the', 3: 'to', 4: 'of', 5: 'and', 6: 'a', 7: 'in', 8: 'that', 9: 'is'
    # print(input_word2idx)   #{'': 90, 'increase': 795, 'deal.': 3431, 'loretta': 2436, 'date': 2724, 'contenders': 3308,

    target_word2idx = dict()
    for idx, word in enumerate(target_counter.most_common(MAX_TARGET_VOCAB_SIZE)):
        target_word2idx[word[0]] = idx + 1
    target_word2idx['UNK'] = 0
    target_idx2word = dict([(idx, word) for word, idx in target_word2idx.items()])

    # print(target_idx2word)   #{0: 'UNK', 1: 'START', 2: 'END', 3: 'the', 4: 'to', 5: 'in', 6: 'of', 7: 'for', 8: 'trump', 9: 'on',
    # print(target_word2idx)   #{{'': 66, 'chance': 870, 'strange': 1891, 'lgbt': 1215, 'saudi': 234, 'armed': 862,

    num_input_tokens = len(input_word2idx)   #
    num_target_tokens = len(target_word2idx)

    config = dict()
    config['input_word2idx'] = input_word2idx
    config['input_idx2word'] = input_idx2word
    config['target_word2idx'] = target_word2idx
    config['target_idx2word'] = target_idx2word
    config['num_input_tokens'] = num_input_tokens
    config['num_target_tokens'] = num_target_tokens
    config['max_input_seq_length'] = max_input_seq_length       #训练集中左右文章中内容单词最多的长度
    config['max_target_seq_length'] = max_target_seq_length        #训练集中左右文章中标题单词最多的长度

    return config


def data_fit(X,Y):
    x,y=[],[]
    for i in range(0,len(X)):
        if len([i]) > 0 and len([i]) > 0:
            seg_list = jieba.cut(Y[i])  # 默认是精确模式
            cont = " ".join(seg_list)
            y.append(data_preprocess(cont.split()))

            seg_list = jieba.cut(X[i])  # 默认是精确模式
            cont = " ".join(seg_list)
            x.append(data_preprocess(cont.split()))

    return fit_text(x,y)