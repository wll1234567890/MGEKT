# -*- coding: utf-8 -*-
# @Time : 2022/4/28 19:18
# @Author : Yumo
# @File : Constant.py
# @Project: GOODKT
# @Comment :
Dpath = '../../Dataset'
datasets = {
    'assist2017' : 'assist2017',
}

# question number of each dataset
numbers = {
    'assist2017' : 3162,
}

skill = {
    'assist2017' : 102,
}


DATASET = datasets['assist2017']
NUM_OF_QUESTIONS = numbers['assist2017']
H = '2017'

MAX_STEP = 50
BATCH_SIZE = 32
LR = 0.0001
EPOCH = 20
EMB = 256
HIDDEN = 128  # sequence model's
kd_loss = 5.00E-06
LAYERS = 1
