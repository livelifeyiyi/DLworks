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
np.savetxt(ROOT_DIR+"wv.txt", wv)


train_data, test_data, dev_data = dm.data['train'], dm.data['test'], dm.data['dev']
random.shuffle(train_data)

train_sentences_words = []  # sentence with words
train_sentences_id = []  # sentence with word_ids
train_entity_tags = []  # entity tags
train_relation_tags = []  # relation type_id
train_relation_names = []  # relation name

print("Processing training data......")
for data in train_data:
	for relation in data["relations"]:  # if sentence has multiple relations
		train_sentences_words.append(data["sentext"])
		train_sentences_id.append(data["text"])
		train_entity_tags.append(relation['tags'])
		train_relation_tags.append(relation['type'])
		train_relation_names.append(relation['rtext'])

print("Writing training data......")
with open(ROOT_DIR+'data_train.pkl', 'wb') as outp:
	pickle.dump(train_sentences_words, outp)
	pickle.dump(train_sentences_id, outp)
	pickle.dump(train_entity_tags, outp)
	pickle.dump(train_relation_tags, outp)
	pickle.dump(train_relation_names, outp)

test_sentences_words = []  # sentence with words
test_sentences_id = []  # sentence with word_ids
test_entity_tags = []  # entity tags
test_relation_tags = []  # relation type_id
test_relation_names = []  # relation name

print("Processing test data......")
for data in test_data:
	for relation in data["relations"]:  # if sentence has multiple relations
		test_sentences_words.append(data["sentext"])
		test_sentences_id.append(data["text"])
		test_entity_tags.append(relation['tags'])
		test_relation_tags.append(relation['type'])
		test_relation_names.append(relation['rtext'])

print("Writing test data......")
with open(ROOT_DIR+'data_test.pkl', 'wb') as outp:
	pickle.dump(test_sentences_words, outp)
	pickle.dump(test_sentences_id, outp)
	pickle.dump(test_entity_tags, outp)
	pickle.dump(test_relation_tags, outp)
	pickle.dump(test_relation_names, outp)

dev_sentences_words = []  # sentence with words
dev_sentences_id = []  # sentence with word_ids
dev_entity_tags = []  # entity tags
dev_relation_tags = []  # relation type_id
dev_relation_names = []  # relation name

print("Processing dev data......")
for data in dev_data:
	for relation in data["relations"]:  # if sentence has multiple relations
		dev_sentences_words.append(data["sentext"])
		dev_sentences_id.append(data["text"])
		dev_entity_tags.append(relation['tags'])
		dev_relation_tags.append(relation['type'])
		dev_relation_names.append(relation['rtext'])

print("Writing dev data......")
with open(ROOT_DIR+'data_dev.pkl', 'wb') as outp:
	pickle.dump(dev_sentences_words, outp)
	pickle.dump(dev_sentences_id, outp)
	pickle.dump(dev_entity_tags, outp)
	pickle.dump(dev_relation_tags, outp)
	pickle.dump(dev_relation_names, outp)
