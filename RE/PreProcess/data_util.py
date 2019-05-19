import codecs
import json
import pickle
import random
import numpy as np
from TFgirl.RE.PreProcess.data_manager import DataManager
from TFgirl.RE.Parser import Parser
import sys

ROOT_DIR = "E:\\newFolder\\data\\entity&relation_dataset\\NYT10\\"
# ROOT_DIR = "C:/(O_O)!/thesis/5-RE with LSTM/code/HRL-RE-use/data/NYT10_demo/"


# argv = sys.argv[1:]
# parser = Parser().getParser()
# args, _ = parser.parse_known_args(argv)
dm = DataManager(ROOT_DIR, 'test')

wv = dm.vector
print(wv.shape)
# np.savetxt(ROOT_DIR+"wv.txt", wv)


train_data, test_data, dev_data = dm.data['train'], dm.data['test'], dm.data['dev']
random.shuffle(train_data)

train_sentences_words = []  # sentence with words
train_sentences_id = []  # sentence with word_ids
train_entity_tags = []  # entity tags
train_relation_tags = []  # relation type_id
train_relation_names = []  # relation name
train_position_lambda = []

print("Processing training data......")
for data in train_data:
	relations = []
	for relation in data["relations"]:
		relations.append(relation['type'])
	for relation in data["relations"]:  # if sentence has multiple relations
		train_sentences_words.append(data["sentext"])
		train_sentences_id.append(data["text"])
		tags = relation['tags']
		train_entity_tags.append(tags)
		train_relation_tags.append(relations)  # (relation['type'])
		train_relation_names.append(relation['rtext'])

		train_position_lambda_i = []
		if 4 in tags and 5 in tags:
			idx_4 = tags.index(4)
			idx_5 = tags.index(5)
			len_d = []
			len_d.append(abs(0-idx_4))
			len_d.append(abs(len(tags)-1-idx_4))
			if idx_4 < idx_5:
				len_d.append(abs(idx_5-1-idx_4))
			else:
				len_d.append(abs(idx_4 - 1 - idx_5))
			len_d.append(abs(0-idx_5))
			len_d.append(abs(len(tags)-1-idx_5))
			D = float(max(len_d))

			for id in range(len(tags)):
				train_position_lambda_i.append((1 - abs(idx_4-id) / D) + (1 - abs(idx_5-id) / D))
		else:
			for id in range(len(tags)):
				train_position_lambda_i.append(0)
		train_position_lambda.append(train_position_lambda_i)


print("Writing training data......")
with open(ROOT_DIR+'data_train.pkl', 'wb') as outp:
	pickle.dump(train_sentences_words, outp)
	pickle.dump(train_sentences_id, outp)
	pickle.dump(train_position_lambda, outp)
	pickle.dump(train_entity_tags, outp)
	pickle.dump(train_relation_tags, outp)
	pickle.dump(train_relation_names, outp)

test_sentences_words = []  # sentence with words
test_sentences_id = []  # sentence with word_ids
test_entity_tags = []  # entity tags
test_relation_tags = []  # relation type_id
test_relation_names = []  # relation name
test_position_lambda = []

print("Processing test data......")
for data in test_data:
	relations = []
	for relation in data["relations"]:
		relations.append(relation['type'])
	for relation in data["relations"]:  # if sentence has multiple relations
		test_sentences_words.append(data["sentext"])
		test_sentences_id.append(data["text"])
		tags = relation['tags']
		test_entity_tags.append(tags)
		test_relation_tags.append(relations)
		test_relation_names.append(relation['rtext'])

		test_position_lambda_i = []
		if 4 in tags and 5 in tags:
			idx_4 = tags.index(4)
			idx_5 = tags.index(5)
			len_d = []
			len_d.append(abs(0 - idx_4))
			len_d.append(abs(len(tags) - 1 - idx_4))
			if idx_4 < idx_5:
				len_d.append(abs(idx_5 - 1 - idx_4))
			else:
				len_d.append(abs(idx_4 - 1 - idx_5))
			len_d.append(abs(0 - idx_5))
			len_d.append(abs(len(tags) - 1 - idx_5))
			D = float(max(len_d))

			for id in range(len(tags)):
				test_position_lambda_i.append((1 - abs(idx_4 - id) / D) + (1 - abs(idx_5 - id) / D))
		else:
			for id in range(len(tags)):
				test_position_lambda_i.append(0)
		test_position_lambda.append(test_position_lambda_i)

print("Writing test data......")
with open(ROOT_DIR+'data_test.pkl', 'wb') as outp:
	pickle.dump(test_sentences_words, outp)
	pickle.dump(test_sentences_id, outp)
	pickle.dump(test_position_lambda, outp)
	pickle.dump(test_entity_tags, outp)
	pickle.dump(test_relation_tags, outp)
	pickle.dump(test_relation_names, outp)

dev_sentences_words = []  # sentence with words
dev_sentences_id = []  # sentence with word_ids
dev_entity_tags = []  # entity tags
dev_relation_tags = []  # relation type_id
dev_relation_names = []  # relation name
dev_position_lambda = []
print("Processing dev data......")
for data in dev_data:
	relations = []
	for relation in data["relations"]:
		relations.append(relation['type'])
	for relation in data["relations"]:  # if sentence has multiple relations
		dev_sentences_words.append(data["sentext"])
		dev_sentences_id.append(data["text"])
		tags = relation['tags']
		dev_entity_tags.append(tags)
		dev_relation_tags.append(relations)
		dev_relation_names.append(relation['rtext'])

		dev_position_lambda_i = []
		if 4 in tags and 5 in tags:
			idx_4 = tags.index(4)
			idx_5 = tags.index(5)
			len_d = []
			len_d.append(abs(0 - idx_4))
			len_d.append(abs(len(tags) - 1 - idx_4))
			if idx_4 < idx_5:
				len_d.append(abs(idx_5 - 1 - idx_4))
			else:
				len_d.append(abs(idx_4 - 1 - idx_5))
			len_d.append(abs(0 - idx_5))
			len_d.append(abs(len(tags) - 1 - idx_5))
			D = float(max(len_d))

			for id in range(len(tags)):
				dev_position_lambda_i.append((1 - abs(idx_4 - id) / D) + (1 - abs(idx_5 - id) / D))
		else:
			for id in range(len(tags)):
				dev_position_lambda_i.append(0)
		dev_position_lambda.append(dev_position_lambda_i)

print("Writing dev data......")
with open(ROOT_DIR+'data_dev.pkl', 'wb') as outp:
	pickle.dump(dev_sentences_words, outp)
	pickle.dump(dev_sentences_id, outp)
	pickle.dump(dev_position_lambda, outp)
	pickle.dump(dev_entity_tags, outp)
	pickle.dump(dev_relation_tags, outp)
	pickle.dump(dev_relation_names, outp)
