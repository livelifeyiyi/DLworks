# coding=utf-8
import codecs
import pandas as pd
import numpy as np
import pickle
import ast

# ROOT_DIR = "E:/code/RE/NYT_processed/"
ROOT_DIR = "NYT/"
tiny_data_number = 100

max_len_train = 374

with codecs.open(ROOT_DIR + 'sentence_train.txt', 'r', encoding='utf-8') as train_xfile:
	train_all_str = train_xfile.readlines()[0:tiny_data_number]
	print(len(train_all_str))
	train_all = []
	for line in train_all_str:
		line = ast.literal_eval(line)
		train_all += line
	print(len(train_all))
with codecs.open(ROOT_DIR + 'sentence_test.txt', 'r', encoding='utf-8') as test_xfile:
	test_all_str = test_xfile.readlines()[0:tiny_data_number]
	test_all = []
	for line in test_all_str:
		line = ast.literal_eval(line)
		test_all += line

whole_ = train_all + test_all
'''words = []
for p in whole_:
	words += p'''
all_words = list(set(whole_))
all_words.append("UKN")
all_words = pd.Series(all_words)
print(len(all_words))
set_ids = all_words.index  # .values
set_words = all_words.values
word2id = pd.Series(set_ids, index=set_words)
id2word = pd.Series(set_words, index=set_ids)

with codecs.open(ROOT_DIR + 'seq_train.txt', 'r', encoding='utf-8') as train_yfile:
	seq_all_str = train_yfile.readlines()[0:tiny_data_number]
	seq_all = []
	for line in seq_all_str:
		line = ast.literal_eval(line)
		seq_all += line
with codecs.open(ROOT_DIR + 'seq_test.txt', 'r', encoding='utf-8') as test_yfile:
	test_yall_str = test_yfile.readlines()[0:tiny_data_number]
	test_yall = []
	for line in test_yall_str:
		line = ast.literal_eval(line)
		test_yall += line

tags_ = seq_all + test_yall
'''tags = []
for q in tags_:
	tags += q'''
tag_dict = list(set(tags_))
tag_dict.append("UKN")
id2tag = pd.Series(tag_dict)
tag_ids = id2tag.index
tag_words = id2tag.values
tag2id = pd.Series(tag_ids, index=tag_words)

print("Processing training sentences......")
train_x = []  # sentences with words'id
index_i = 0
for line in train_all_str:
	line = ast.literal_eval(line)
	line_x = []
	# index_j = 0
	for word in line:
		word_id = word2id[word]
		line_x.append(word_id)
	line_x = np.pad(line_x, (0, max_len_train - len(line_x)), 'constant', constant_values=(0, word2id["UKN"]))
	train_x.append(line_x)
	print(index_i)
	index_i += 1

print("Processing training sequences......")
train_y = []
index_i = 0
for line in seq_all_str:
	line = ast.literal_eval(line)
	line_y = []
	for word in line:
		line_y.append(tag2id[word])
	line_y = np.pad(line_y, (0, max_len_train - len(line_y)), 'constant', constant_values=(0, tag2id["UKN"]))
	train_y.append(line_y)
	index_i += 1

print("Writing train_y......")
with open(ROOT_DIR + 'RE_data_train_pad_tiny.pkl', 'wb') as outp:
	pickle.dump(word2id, outp)
	pickle.dump(id2word, outp)
	pickle.dump(tag2id, outp)
	pickle.dump(train_x, outp)
	pickle.dump(train_y, outp)
print('** Finished saving train data.')

del train_x
del train_y
max_len_test = 131
print("Processing test sentences......")
test_x = []
index_i = 0
for line in test_all_str:
	line = ast.literal_eval(line)
	line_x = []
	for word in line:
		if "\r\n" in word:
			word = "."
		word_id = word2id[word]
		line_x.append(word_id)
	line_x = np.pad(line_x, (0, max_len_test - len(line_x)), 'constant', constant_values=(0, word2id["UKN"]))
	test_x.append(line_x)
	index_i += 1
print("Processing test sequences......")
test_y = []
index_i = 0
for line in test_yall_str:
	line = ast.literal_eval(line)
	line_y = []
	for word in line:
		line_y.append(tag2id[word])
	line_y = np.pad(line_y, (0, max_len_test - len(line_y)), 'constant', constant_values=(0, tag2id["UKN"]))
	test_y.append(line_y)
	index_i += 1

with open(ROOT_DIR + 'RE_data_test_pad_tiny.pkl', 'wb') as outp:
	pickle.dump(test_x, outp)
	pickle.dump(test_y, outp)
print('** Finished saving test data.')
