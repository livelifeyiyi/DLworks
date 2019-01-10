# coding=utf-8
import codecs
import sys
import re
import pandas as pd
import numpy as np
import pickle
import ast
from collections import deque

ROOT_DIR = "E:/code/RE/NYT_processed/"
# ROOT_DIR = "NYT/"

with codecs.open(ROOT_DIR + 'word_dict.txt', 'r', encoding='utf-8') as word_file:
	all_words = word_file.read().splitlines()
	print(len(all_words))

all_words = pd.Series(all_words)
# all_words = pd.read_csv(ROOT_DIR + "word_dict.txt", sep='\s+', index_col=None, encoding="utf-8")  # Series.from_csv
# pd.read_table(ROOT_DIR+"word_dict.txt")
set_ids = all_words.index  # .values
set_words = all_words.values
word2id = pd.Series(set_ids, index=set_words)
id2word = pd.Series(set_words, index=set_ids)

id2tag = pd.Series.from_csv(ROOT_DIR + "tag_dict.txt", sep=' ', index_col=None, encoding="utf-8")
tag_ids = id2tag.index
tag_words = id2tag.values
tag2id = pd.Series(tag_ids, index=tag_words)

print("Processing training sequences......")
train_y = []
with codecs.open(ROOT_DIR + 'seq_train.txt', 'r', encoding='utf-8') as train_yfile:
	seq_all = train_yfile.readlines()
	for line in seq_all:
		line = ast.literal_eval(line)
		line_y = []
		for word in line:
			line_y.append(tag2id[word])
		train_y.append(line_y)

print("Processing training sentences......")
train_x = []  # sentences with words'id
with codecs.open(ROOT_DIR + 'sentence_train.txt', 'r', encoding='utf-8') as train_xfile:
	train_all = train_xfile.readlines()
	for line in train_all:
		line = ast.literal_eval(line)
		line_x = []
		for word in line:
			line_x.append(word2id[word])
		train_x.append(line_x)

with open(ROOT_DIR + 'RE_data_train.pkl', 'wb') as outp:
	pickle.dump(word2id, outp)
	pickle.dump(id2word, outp)
	pickle.dump(tag2id, outp)
	pickle.dump(train_x, outp)
	pickle.dump(train_y, outp)
# pickle.dump(pos_e1, outp)
# pickle.dump(pos_e2, outp)
print('** Finished saving train data.')

test_x = []
with codecs.open(ROOT_DIR + 'sentence_test.txt', 'r', encoding='utf-8') as test_xfile:
	test_all = test_xfile.readlines()
	for line in test_all:
		line_x = []
		for word in line:
			line_x.append(word2id[word])
		test_x.append(line_x)
test_y = []
with codecs.open(ROOT_DIR + 'seq_test.txt', 'r', encoding='utf-8') as test_yfile:
	test_yall = test_yfile.readlines()
	for line in test_yall:
		line_y = []
		for word in line:
			line_y.append(word2id[word])
		test_y.append(line_y)

with open(ROOT_DIR + 'RE_data_test.pkl', 'wb') as outp:
	pickle.dump(test_x, outp)
	pickle.dump(test_y, outp)
print('** Finished saving test data.')
