# coding:utf8
import numpy as np
import pickle
import sys
import codecs
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
from torch.autograd import Variable
import torch.optim as optim

torch.manual_seed(1)
'''
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda:0")  # device = torch.device("cpu")
ROOT_DIR = "./NYT/"
# model_PATH = "./model/model_LSTMCRF.pkl"
# torch.set_default_tensor_type('torch.cuda.DoubleTensor') # torch.set_default_tensor_type('torch.DoubleTensor')
'''
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ROOT_DIR = "./NYT/"
ROOT_DIR = "C:\\(O_O)!\\thesis\\5-RE with LSTM\\code\\testData\\"

VECTOR_NAME = ROOT_DIR+"vec.txt"


class BiLSTM_ATT(nn.Module):
	def __init__(self, config, embedding_pre):
		super(BiLSTM_ATT, self).__init__()
		self.batch = config['BATCH']

		self.embedding_size = config['EMBEDDING_SIZE']
		self.embedding_dim = config['EMBEDDING_DIM']

		self.hidden_dim = config['HIDDEN_DIM']
		self.tag_size = config['TAG_SIZE']

		# self.pos_size = config['POS_SIZE']
		# self.pos_dim = config['POS_DIM']

		self.pretrained = config['pretrained']
		if self.pretrained:
			# self.word_embeds.weight.data.copy_(torch.from_numpy(embedding_pre))
			self.word_embeds = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_pre), freeze=False)
		else:
			self.word_embeds = nn.Embedding(self.embedding_size, self.embedding_dim)

		# self.pos1_embeds = nn.Embedding(self.pos_size, self.pos_dim)
		# self.pos2_embeds = nn.Embedding(self.pos_size, self.pos_dim)
		self.relation_embeds = nn.Embedding(self.tag_size, self.hidden_dim)
		self.bilstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim // 2,
							num_layers=1, bidirectional=True)  #  + self.pos_dim * 2
		self.lstm = nn.LSTM(self.embedding_dim*2, self.hidden_dim, num_layers=1)
		self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)

		self.dropout_emb = nn.Dropout(p=0.5)
		self.dropout_lstm = nn.Dropout(p=0.5)
		self.dropout_att = nn.Dropout(p=0.5)

		self.hidden = self.init_hidden()
		self.att_weight = nn.Parameter(torch.randn(self.batch, 1, self.hidden_dim, device=device))
		self.relation_bias = nn.Parameter(torch.randn(self.batch, self.tag_size, 1, device=device))

	def init_hidden(self):
		return torch.randn(2, self.batch, self.hidden_dim // 2, device=device)

	def init_hidden_lstm(self):
		return (torch.randn(2, self.batch, self.hidden_dim // 2, device=device),
				torch.randn(2, self.batch, self.hidden_dim // 2, device=device))

	def attention(self, H):
		M = torch.tanh(H)
		a = F.softmax(torch.bmm(self.att_weight, M), 2)
		a = torch.transpose(a, 1, 2)
		return torch.bmm(H, a)

	def forward(self, sentences):  # , pos1, pos2
		if torch.cuda.is_available():
			sentences = Variable(torch.cuda.LongTensor(sentences))
		else:
			sentences = Variable(torch.LongTensor(sentences))
		self.hidden = self.init_hidden_lstm()
		# embeds = torch.cat((self.word_embeds(sentence)), 2)  # , self.pos1_embeds(pos1), self.pos2_embeds(pos2)
		embeds = self.word_embeds(sentences)
		embeds = torch.transpose(embeds, 0, 1)
		lstm_out, self.hidden = self.bilstm(embeds, self.hidden)
		# lstm_out = torch.transpose(lstm_out, 0, 1)
		# lstm_out = torch.transpose(lstm_out, 1, 2)
		# lstm_out = self.dropout_lstm(lstm_out)
		# reshape
		self.hidden = (torch.reshape(torch.cat((self.hidden[0][0], self.hidden[0][1]), 1), (1, self.batch, -1)), torch.reshape(torch.cat((self.hidden[1][0], self.hidden[1][1]), 1), (1, self.batch, -1)))

		# self.hidden = torch.cat((self.hidden[0], self.hidden[1]), 2)
		decode_out, self.hidden = self.lstm(lstm_out, self.hidden)  # self.init_hidden()
		decode_out = torch.transpose(decode_out, 0, 1)
		decode_out = torch.transpose(decode_out, 1, 2)
		decode_out = self.dropout_lstm(decode_out)
		# att_out = torch.tanh(lstm_out)  # self.attention(lstm_out)
		# att_out = self.dropout_att(att_out)
		relation = torch.tensor([i for i in range(self.tag_size)], dtype=torch.long, device=device).repeat(self.batch, 1)
		relation = self.relation_embeds(relation)
		res = torch.add(torch.bmm(relation, decode_out), self.relation_bias)
		res = F.softmax(res, 1)
		return res.view(self.batch, -1)


with open(ROOT_DIR + 'RE_data_train_pad.pkl', 'rb') as inp:
	# with codecs.open(ROOT_DIR+'RE_data_train.pkl', 'rb', encoding="utf-8") as inp:
	word2id = pickle.load(inp, encoding='latin1')
	id2word = pickle.load(inp, encoding='latin1')
	tag2id = pickle.load(inp, encoding='latin1')
	train_x = pickle.load(inp, encoding='latin1')  # train sentence
	train_y = pickle.load(inp, encoding='latin1')  # train sequence

with open(ROOT_DIR + 'RE_data_test_pad.pkl', 'rb') as inp:
	# with codecs.open(ROOT_DIR+'RE_data_test.pkl', 'rb', encoding="utf-8") as inp:
	test_x = pickle.load(inp, encoding='latin1')  # test sentence
	test_y = pickle.load(inp, encoding='latin1')  # test sequence
# position1_t = pickle.load(inp)
# position2_t = pickle.load(inp)

print("train len", len(train_x))
print("test len", len(test_x))
print("word2id len", len(word2id))

EMBEDDING_SIZE = len(word2id) + 1
EMBEDDING_DIM = 300

# POS_SIZE = 82  # 不同数据集这里可能会报错。
# POS_DIM = 25

HIDDEN_DIM = 600  # 300

TAG_SIZE = len(tag2id)

BATCH = 64  # 128  # 100
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
criterion = nn.CrossEntropyLoss()
loss_his = []

if torch.cuda.is_available():
	model = model.cuda()
	criterion = criterion.cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

'''x_len = []
train_x_batch = train_x[:len(train_x) - len(train_x) % BATCH]
for x in train_x_batch: x_len.append(len(x))
max_length = max(x_len)'''

if torch.cuda.is_available():
	train_x = torch.cuda.LongTensor(train_x[:len(train_x) - len(train_x) % BATCH], device=device)  # .cuda()
	train_y = torch.cuda.LongTensor(train_y[:len(train_x) - len(train_x) % BATCH], device=device)  # .cuda()
else:
	train_x = torch.LongTensor(train_x, device=device)  # .cuda()
	train_y = torch.LongTensor(train_y, device=device)  # .cuda()
# position1 = torch.LongTensor(position1[:len(train) - len(train) % BATCH])
# position2 = torch.LongTensor(position2[:len(train) - len(train) % BATCH])
train_datasets = D.TensorDataset(train_x, train_y)  # , position1, position2
train_dataloader = D.DataLoader(train_datasets, BATCH, True)  # , num_workers=2

if torch.cuda.is_available():
	test_x = torch.cuda.LongTensor(test_x[:len(test_x) - len(test_x) % BATCH], device=device)
	test_y = torch.cuda.LongTensor(test_y[:len(test_x) - len(test_x) % BATCH], device=device)
else:
	test_x = torch.LongTensor(test_x, device=device)
	test_y = torch.LongTensor(test_y, device=device)
# position1_t = torch.LongTensor(position1_t[:len(test) - len(test) % BATCH])
# position2_t = torch.LongTensor(position2_t[:len(test) - len(test) % BATCH])
test_datasets = D.TensorDataset(test_x, test_y)  # , position1_t, position2_t
test_dataloader = D.DataLoader(test_datasets, BATCH, True)  # , num_workers=2

for epoch in range(EPOCHS):
	print("epoch:", epoch)
	acc = 0
	total = 0
	# for step, (sentence_in, targets) in enumerate(train_dataloader):
	for sentence, tag in train_dataloader:  # , pos1, pos2
		# sentence = Variable(sentence)
		# pos1 = Variable(pos1)
		# pos2 = Variable(pos2)
		y_hat = model(sentence)  # , pos1, pos2
		if torch.cuda.is_available():
			tags = Variable(tag).cuda()
		else:
			tags = Variable(tag)
		# print(y_hat)
		# print(tags)
		tags = tags.argmax(1)
		loss = criterion(y_hat, tags)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print(loss)
		loss_his.append(loss.data)
		# np.savetxt('loss.during', loss_his)
		# y_hat = np.argmax(y_hat.data, axis=1)
		y_hat = y_hat.argmax(1)

		'''for y1, y2 in zip(y_hat, tags):
			if y1 == y2:
				acc += 1
			total += 1'''

	# print("train:", 100 * float(acc) / total, "%")
	np.savetxt('loss.during', loss_his)
	'''acc_t = 0
	total_t = 0
	count_predict = [0 for i in range(len(tag2id))]
	count_total = [0 for i in range(len(tag2id))]
	count_right = [0 for i in range(len(tag2id))]
	for sentence, tag in test_dataloader:  # , pos1, pos2
		sentence = Variable(sentence)
		# pos1 = Variable(pos1)
		# pos2 = Variable(pos2)
		y_hat = model(sentence)  # , pos1, pos2
		# y_hat = np.argmax(y_hat.data, axis=1)
		y_hat = y_hat.argmax(1)
		tag = tag.argmax(1)
		for y1, y2 in zip(y_hat, tag):
			count_predict[y1] += 1
			count_total[y2] += 1
			if y1 == y2:
				count_right[y1] += 1

	precision = [0 for i in range(len(tag2id))]
	recall = [0 for i in range(len(tag2id))]
	for i in range(len(count_predict)):
		if count_predict[i] != 0:
			precision[i] = float(count_right[i]) / count_predict[i]

		if count_total[i] != 0:
			recall[i] = float(count_right[i]) / count_total[i]

	precision = sum(precision) / len(tag2id)
	recall = sum(recall) / len(tag2id)
	print("准确率：", precision)
	print("召回率：", recall)
	print("f：", (2 * precision * recall) / (precision + recall))'''

	if epoch % 10 == 0:
		model_name = "./model/model_BiLSTM_LSTM_epoch" + str(epoch) + ".pkl"
		torch.save(model, model_name)
		print(model_name, "has been saved")

torch.save(model, "./model/model_BiLSTM_LSTM.pkl")
print("model has been saved")
