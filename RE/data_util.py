# coding=utf-8
import pickle
import random
import unicodedata
import re
import json
import codecs

SOS_token = 0
EOS_token = 1

ROOT_DIR = "E:\\newFolder\\data\\entity&relation_dataset\\NYT10\\"


class Lang:
	def __init__(self, name):
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0: "SOS", 1: "EOS"}
		self.n_words = 2  # Count SOS and EOS

	def addSentence(self, sentence):
		for word in sentence.split(' '):
			self.addWord(word)

	def addWord(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1
'''
sentence_len =[]
with codecs.open(ROOT_DIR+"train.json", "r", encoding="utf-8") as f:
	for line in f.readlines():
		sentence_json = json.loads(line)
		sentence = sentence_json["sentext"]
		sentence_len.append(len(sentence.split()))
print(sentence_len)
print(max(sentence_len))
'''

def data2pkl():
	print("Processing training data......")
	input = Lang("sentence")
	tag2id = {}

	train_y = []
	with codecs.open(ROOT_DIR+"train.json", "r", encoding="utf-8") as f:
		for line in f.readlines():
			sentence_json = json.loads(line)
			sentence = sentence_json["sentext"]
			tag = sentence_json["relations"][0]["tags"]
			input.addSentence(sentence)
			train_y.append(tag)
			for t in tag:
				if tag2id == {}:
					tag2id[t] = 0
				if t in tag2id.keys():
					pass
				else:
					tag2id[t] = len(tag2id)
	print(tag2id)

	train_x = []
	with codecs.open(ROOT_DIR+"train.json", "r", encoding="utf-8") as f:
		for line in f.readlines():
			sentence_json = json.loads(line)
			sentence = sentence_json["sentext"]
			indexes = [input.word2index[word] for word in sentence.split(' ')]
			indexes.append(EOS_token)
			train_x.append(indexes)
	# word2id = input.word2index
	id2word = input.index2word
	with open(ROOT_DIR+'RE_data_train.pkl', 'wb') as outp:
		pickle.dump(id2word, outp)
		pickle.dump(train_x, outp)
		pickle.dump(train_y, outp)

	print("Processing test data......")
	test_sent = Lang("sentence")

	test_y = []
	with codecs.open(ROOT_DIR+"test.json", "r", encoding="utf-8") as f:
		for line in f.readlines():
			sentence_json = json.loads(line)
			sentence = sentence_json["sentext"]
			tag = sentence_json["relations"][0]["tags"]
			test_sent.addSentence(sentence)
			test_y.append(tag)

	test_x = []
	with codecs.open(ROOT_DIR+"test.json", "r", encoding="utf-8") as f:
		for line in f.readlines():
			sentence_json = json.loads(line)
			sentence = sentence_json["sentext"]
			indexes = [test_sent.word2index[word] for word in sentence.split(' ')]
			indexes.append(EOS_token)
			test_x.append(indexes)

	word2id_test = test_sent.word2index
	with open(ROOT_DIR+'RE_data_test.pkl', 'wb') as outp:
		pickle.dump(word2id_test, outp)
		pickle.dump(test_x, outp)
		pickle.dump(test_y, outp)

data2pkl()
def get_one_random_pair():
	"""
	随机挑选一个 sentence-tag对，返回其对应的index
	"""
	def readLangs():
		with open("train.json", "r") as f:
			data = json.load(f)
		pairs = []
		for i in range(len(data)):
			pair = []
			sentencee = data[i]["sentext"]
			tag = data[i]["tags"]
			pair.append(sentencee)
			pair.append(tag)
			pairs.append(pair)
		input = Lang("sentence")
		output = Lang("tag")
		return input, output, pairs

	def prepareData():
		input_lang, output_lang, pairs = readLangs()
		print("Read %s sentence pairs" % len(pairs))
		print("Counting words...")
		for pair in pairs:
			input_lang.addSentence(pair[0])
			output_lang.addSentence(pair[1])
		print("Counted words:")
		print(input_lang.name, input_lang.n_words)
		print(output_lang.name, output_lang.n_words)
		return input_lang, output_lang, pairs

	def indexesFromSentence(lang, sentence):
		return [lang.word2index[word] for word in sentence.split(' ')]

	def tensorFromSentence(lang, sentence):
		indexes = indexesFromSentence(lang, sentence)
		indexes.append(EOS_token)
		return indexes  # torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

	def tensorsFromPair(pair):
		input_indexes = tensorFromSentence(input_lang, pair[0])
		target_indexes = tensorFromSentence(output_lang, pair[1])
		return (input_indexes, target_indexes)

	input_lang, output_lang, pairs = prepareData()
	tensorsFromPair(random.choice(pairs))