# coding:utf8
import numpy as np
import pickle
import sys
import codecs
# ROOT_DIR = "E:/code/RE/NYT_processed/"
ROOT_DIR = "NYT/"
VECTOR_NAME = ROOT_DIR+"vec.txt"
# with open('./data/engdata_train.pkl', 'rb') as inp:
with open(ROOT_DIR+'RE_data_train.pkl', 'rb') as inp:
	word2id = pickle.load(inp)
	id2word = pickle.load(inp)
	tag2id = pickle.load(inp)
	train_x = pickle.load(inp)  # train sentence
	train_y = pickle.load(inp)  # train sequence
	
# with open('./data/engdata_test.pkl', 'rb') as inp: 
with open(ROOT_DIR+'RE_data_test.pkl', 'rb') as inp:
	test_x = pickle.load(inp)  # test sentence
	test_y = pickle.load(inp)  # test sequence
	# position1_t = pickle.load(inp)
	# position2_t = pickle.load(inp)

print("train len", len(train_x))
print("test len", len(test_x))
print("word2id len", len(word2id))

import torch
import torch.nn as nn
import torch.optim as optim

import torch.utils.data as D
from torch.autograd import Variable
from BiLSTM_ATT import BiLSTM_ATT

EMBEDDING_SIZE = len(word2id) + 1
EMBEDDING_DIM = 100

# POS_SIZE = 82  # 不同数据集这里可能会报错。
# POS_DIM = 25

HIDDEN_DIM = 200

TAG_SIZE = len(tag2id)

BATCH = 1000
EPOCHS = 100

config = {}
config['EMBEDDING_SIZE'] = EMBEDDING_SIZE
config['EMBEDDING_DIM'] = EMBEDDING_DIM
# config['POS_SIZE'] = POS_SIZE
# config['POS_DIM'] = POS_DIM
config['HIDDEN_DIM'] = HIDDEN_DIM
config['TAG_SIZE'] = TAG_SIZE
config['BATCH'] = BATCH
config["pretrained"] = False

learning_rate = 0.0005

embedding_pre = []
if len(sys.argv) == 2 and sys.argv[1] == "pretrained":
	print("use pretrained embedding")
	config["pretrained"] = True
	word2vec = {}
	with codecs.open(VECTOR_NAME, 'r', 'utf-8') as input_data:
		for line in input_data.readlines():
			word2vec[line.split()[0]] = map(eval, line.split()[1:])

	unknow_pre = []
	unknow_pre.extend([1] * 100)
	embedding_pre.append(unknow_pre)  # wordvec id 0
	for word in word2id:
		if word2vec.has_key(word):
			embedding_pre.append(word2vec[word])
		else:
			embedding_pre.append(unknow_pre)

	embedding_pre = np.asarray(embedding_pre)
	print(embedding_pre.shape)

model = BiLSTM_ATT(config, embedding_pre)
# model = torch.load('model/model_epoch20.pkl')
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss(size_average=True)

train_x = torch.LongTensor(train_x[:len(train_x) - len(train_x) % BATCH])
# position1 = torch.LongTensor(position1[:len(train) - len(train) % BATCH])
# position2 = torch.LongTensor(position2[:len(train) - len(train) % BATCH])
train_y = torch.LongTensor(train_y[:len(train_x) - len(train_x) % BATCH])
train_datasets = D.TensorDataset(train_x, y)  # , position1, position2
train_dataloader = D.DataLoader(train_datasets, BATCH, True, num_workers=2)

test_x = torch.LongTensor(test_x[:len(test_x) - len(test_x) % BATCH])
# position1_t = torch.LongTensor(position1_t[:len(test) - len(test) % BATCH])
# position2_t = torch.LongTensor(position2_t[:len(test) - len(test) % BATCH])
test_y = torch.LongTensor(test_y[:len(test_x) - len(test_x) % BATCH])
test_datasets = D.TensorDataset(test_x, test_y)  # , position1_t, position2_t
test_dataloader = D.DataLoader(test_datasets, BATCH, True, num_workers=2)

for epoch in range(EPOCHS):
	print("epoch:", epoch)
	acc = 0
	total = 0

	for sentence, tag in train_dataloader:  # , pos1, pos2
		sentence = Variable(sentence)
		# pos1 = Variable(pos1)
		# pos2 = Variable(pos2)
		y_hat = model(sentence)  # , pos1, pos2
		tags = Variable(tag)
		loss = criterion(y_hat, tags)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		y_hat = np.argmax(y_hat.data.numpy(), axis=1)

		for y1, y2 in zip(y_hat, tag):
			if y1 == y2:
				acc += 1
			total += 1

	print("train:", 100 * float(acc) / total, "%")

	acc_t = 0
	total_t = 0
	count_predict = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	count_total = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	count_right = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	for sentence, tag in test_dataloader:  # , pos1, pos2
		sentence = Variable(sentence)
		# pos1 = Variable(pos1)
		# pos2 = Variable(pos2)
		y_hat = model(sentence)  # , pos1, pos2
		y_hat = np.argmax(y_hat.data.numpy(), axis=1)
		for y1, y2 in zip(y_hat, tag):
			count_predict[y1] += 1
			count_total[y2] += 1
			if y1 == y2:
				count_right[y1] += 1

	precision = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	recall = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	for i in range(len(count_predict)):
		if count_predict[i] != 0:
			precision[i] = float(count_right[i]) / count_predict[i]

		if count_total[i] != 0:
			recall[i] = float(count_right[i]) / count_total[i]

	precision = sum(precision) / len(tag2id)
	recall = sum(recall) / len(tag2id)
	print("准确率：", precision)
	print("召回率：", recall)
	print("f：", (2 * precision * recall) / (precision + recall))

	if epoch % 20 == 0:
		model_name = "./model/model_epoch" + str(epoch) + ".pkl"
		torch.save(model, model_name)
		print(model_name, "has been saved")

torch.save(model, "./model/model_01.pkl")
print("model has been saved")
