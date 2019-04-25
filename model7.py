#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2018/12/11 9:35
# @Author  : Jing Zhang
# @Site    : 
# @File    : model7.py
# @Software: PyCharm
from keras.layers import Embedding, Dense, Input, Reshape,Conv1D
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
import jieba
import textfit
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
# import tensorflow as tf
from keras.callbacks import TensorBoard
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
print(tf.__version__)

HIDDEN_UNITS = 100
DEFAULT_BATCH_SIZE = 10
VERBOSE = 1
DEFAULT_EPOCHS = 100



file_x = open("shortX.txt", 'r', encoding='utf-8')
file_y = open("shorty.txt", 'r', encoding='utf-8')
X, Y = [], []
for line_x, line_y in zip(file_x, file_y):
    X.append(line_x)
    Y.append(line_y)

config = textfit.data_fit(X,Y)     #返回分词后，模型config的结果
num_input_tokens = config['num_input_tokens']
max_input_seq_length = config['max_input_seq_length']
num_target_tokens = config['num_target_tokens']
max_target_seq_length = config['max_target_seq_length']
input_word2idx = config['input_word2idx']
input_idx2word = config['input_idx2word']
target_word2idx = config['target_word2idx']
target_idx2word = config['target_idx2word']



def delta_mse(y_true, y_pred):



    return None


def build_model(encoder_inputs,decoder_inputs,decoder_state_inputs):
    encoder_embedding = Embedding(input_dim=num_input_tokens,
                                  output_dim=HIDDEN_UNITS,
                                  input_length=max_input_seq_length,
                                  name='encoder_embedding')
    # encoder层的中间状态 encoder_state_h 和 encoder_state_c
    # The encoder model
    encoder_lstm = LSTM(units=HIDDEN_UNITS,
                        return_state=True,
                        name='encoder_lstm')
    encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedding(encoder_inputs))
    encoder_states = [encoder_state_h, encoder_state_c]

    # The decode
    # Used to generate an state vector in the training period
    decoder_lstm = LSTM(units=HIDDEN_UNITS,
                        return_state=True,
                        return_sequences=True,
                        name='decoder_lstm')
    decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                     initial_state=encoder_states)
    decoder_dense = Dense(units=num_target_tokens,
                          activation='softmax',
                          name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs,decoder_inputs],  decoder_outputs)
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy'])

    encoder_model = Model(encoder_inputs, encoder_states)

    # The decoder model
    # Used to generate the outputs in prediction period
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs,initial_state=decoder_state_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    return encoder_model,decoder_model,model



def build_discriminator(input):
    b = Conv1D(500,(10),activation='relu',padding='same')(input)
    c = Conv1D(500,(10),activation='relu',padding='same')(b)
    d = Conv1D(num_target_tokens,(10),activation='relu',padding='same')(c)
    model = Model(inputs=input, outputs=d)
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    print(model.summary())
    return model,d

def build_discriminator_1(input):
    b = Conv1D(500,(10),activation='relu',padding='same')(input)
    c = Conv1D(500,(10),activation='relu',padding='same')(b)
    d = Conv1D(num_target_tokens,(10),activation='relu',padding='same')(c)
    model = Model(inputs=input, outputs=d)
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    print(model.summary())
    return model,d

def transform_input_encoding(text):
    """
    :param text: 分词后的词语list
    :return: 根据self.input_word2idx返回对应得词语ID（数字）
    """
    temp = []
    for line in text:
        x = []
        for word in line:
            wid = 1
            if word in input_word2idx:
                wid = input_word2idx[word]
            x.append(wid)
            if len(x) >= max_input_seq_length:
                break
        temp.append(x)
    temp = pad_sequences(temp, maxlen=max_input_seq_length)
    return temp

def transform_target_encoding(texts):
    temp = []
    for line in texts:
        x = []
        line2 = ['START'] + line + ['END']
        for word in line2:
            x.append(word)
            if len(x) >= max_target_seq_length:
                break
        temp.append(x)
    temp = np.array(temp)
    return temp


def generate_data(x_samples, y_samples, batch_size):
    encoder_input_data_batch = pad_sequences(x_samples, max_input_seq_length)
    decoder_target_data_batch = np.zeros(shape=(batch_size, max_target_seq_length, num_target_tokens))
    decoder_input_data_batch = np.zeros(shape=(batch_size, max_target_seq_length, num_target_tokens))
    for lineIdx, target_words in enumerate(y_samples):
        #lineIdx行号，target_words：list
        for idx, w in enumerate(target_words):
            w2idx = 0  # default [UNK]
            if w in target_word2idx:
                w2idx = target_word2idx[w]
            if w2idx != 0:
                decoder_input_data_batch[lineIdx, idx, w2idx] = 1
                if idx > 0:
                    decoder_target_data_batch[lineIdx, idx - 1, w2idx] = 1

    return [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch


def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def make_gan(encoder_in,decoder_in, G, D):
    set_trainability(D, False)
    x = G([encoder_in,decoder_in])
    auto_encoder_out = D(x)
    auto_encoder = Model([encoder_in,decoder_in], auto_encoder_out)
    auto_encoder.compile(loss='categorical_crossentropy', optimizer=G.optimizer)
    return auto_encoder, auto_encoder_out


def pretrain(D, batch_size=2):
    x, y = [], []
    for i in range(0, len(X)):
        if len(X[i]) > 0 and len(Y[i]) > 0:
            seg_list = jieba.cut(Y[i])  # 默认是精确模式
            cont = " ".join(seg_list)
            y.append(textfit.data_preprocess(cont.split()))
            seg_list = jieba.cut(X[i])  # 默认是精确模式
            cont = " ".join(seg_list)
            x.append(textfit.data_preprocess(cont.split()))
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.1, random_state=42)
    inputx = Xtrain[0:batch_size]
    inputy = Ytrain[0:batch_size]
    print(inputx)
    print(inputy)
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    input_x = transform_input_encoding(inputx)
    input_y = transform_target_encoding(inputy)
    x, y = generate_data(input_x, input_y, batch_size)
    x[0] = x[0] * 1.0
    x[1] = x[1] * 1.0
    set_trainability(D, True)
    D.fit(x[1], fake, epochs=1, batch_size=batch_size)


def summarize(input_text, encoder, decoder):
    input_seq = []
    input_wids = []
    input_text = jieba.cut(input_text)  # 默认是精确模式
    input_text = " ".join(input_text)
    for word in input_text.split(' '):
        idx = 1  # default [UNK]
        if word in input_word2idx:
            idx = input_word2idx[word]
        input_wids.append(idx)
    input_seq.append(input_wids)
    input_seq = pad_sequences(input_seq, max_input_seq_length)
    states_value = encoder.predict(input_seq)
    target_seq = np.zeros((1, 1, num_target_tokens))
    target_seq[0, 0, target_word2idx['START']] = 1
    target_text = ''
    target_text_len = 0
    terminated = False
    while not terminated:
        output_tokens, h, c = decoder.predict([target_seq] + states_value)

        sample_token_idx = np.argmax(output_tokens[0, -1, :])
        sample_word = target_idx2word[sample_token_idx]
        target_text_len += 1

        if sample_word != 'START' and sample_word != 'END':
            target_text += ' ' + sample_word

        if sample_word == 'END' or target_text_len >= max_target_seq_length:
            terminated = True

        target_seq = np.zeros((1, 1, num_target_tokens))
        target_seq[0, 0, sample_token_idx] = 1

        states_value = [h, c]
    return target_text.strip()


def train(auto_encoder,discriminator,epochs=10, batch_size=16):
    x, y = [], []
    for i in range(0, len(X)):
        if len(Y[i]) > 0 and len(X[i]) > 0:
            seg_list = jieba.cut(Y[i])  # 默认是精确模式
            cont = " ".join(seg_list)
            y.append(textfit.data_preprocess(cont.split()))
            seg_list = jieba.cut(X[i])  # 默认是精确模式
            cont = " ".join(seg_list)
            x.append(textfit.data_preprocess(cont.split()))

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.1, random_state=42)
    """the len of train data is 105, type list"""
    """the len of test data is 12, type list"""

    log_path = './graph'
    callback = TensorBoard(log_path)
    callback.set_model(auto_encoder)

    d_loss_real,d_loss_fake=[],[]
    g_loss = []
    for epoch in range(epochs):
        print("epochs:",epoch)
        index = 0
        #  Train Discriminator
        while index < (len(Xtrain)-batch_size):
            inputx = Xtrain[index:index+batch_size]
            inputy = Ytrain[index:index+batch_size]
            input_x = transform_input_encoding(inputx)
            input_y= transform_target_encoding(inputy)
            x,y = generate_data(input_x,input_y,batch_size)
            x[0] = x[0]*1.0
            x[1] = x[1]*1.0
            set_trainability(discriminator, True)
            d_loss_real.append(discriminator.train_on_batch(x[1], y))

            set_trainability(discriminator, False)
            g_loss.append(auto_encoder.train_on_batch(x, y))
            # write_log(callback, train_names, g_loss[-1], index)
            index+=batch_size
            print(d_loss_real[-1],"\t",g_loss[-1])
    return d_loss_real,d_loss_fake, g_loss


en_input = Input(shape=(None,), name='en_input')
de_input = Input(shape=(None, num_target_tokens), name='de_input')
decoder_state_inputs = [Input(shape=(HIDDEN_UNITS,)), Input(shape=(HIDDEN_UNITS,))]
encoder,decoder,G = build_model(en_input, de_input, decoder_state_inputs)
# G.summary()

D_in = Input(shape=(None, num_target_tokens), name='dis_inputs')
D, D_out = build_discriminator(D_in)
D.summary()

GAN, GAN_out = make_gan(en_input,de_input, G, D)
# GAN.summary()


# pretrain(D)

d_loss_real,d_loss_fake, g_loss = train(GAN, D,epochs=100)
# for layer in GAN.layers:
#     if layer.name in ['model_1','model_4']:
#         print(layer.summary())

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.1, random_state=42)
# for i in Xtest:
#     print(i)
for i in range(10):
    title = summarize(Xtrain[i],encoder,decoder)
    print(i,Ytrain[i])
    print(i,title)
print('---------------------我是分割线------------------------')
for i in range(10):
    title = summarize(Xtest[i],encoder,decoder)
    print(i,Ytest[i])
    print(i,title)