# to process the dataset into tagged format which fit for the model.
import pickle
import random

import codecs

import numpy as np

from TFgirl.RE.PreProcess.data_manager import DataManager

# =============================================global parameters and hyperparameters==================================
# EMBEDDING = 300
# DROPOUT = 0.5
# LSTM_ENCODE = 300
# LSTM_DECODE = 600
# BIAS_ALPHA = 10
# VALIDATION_SIZE = 0.1
# EPOCH_NUM = 1000
# # "E:\\data\\entity&relation_dataset\\NYT\\"
# FILE_PATH = './NYT/'
# TRAIN_PATH = FILE_PATH + 'train.json'
# TEST_PATH = FILE_PATH + 'test.json'
# """
# X_TRAIN = '/home/aistudio/data/data1272/sentence_train.txt'
# Y_TRAIN = '/home/aistudio/data/data1272/seq_train.txt'
# X_TEST = '/home/aistudio/data/data1272/sentence_test.txt'
# Y_TEST = '/home/aistudio/data/data1272/seq_test.txt'
# WORD_DICT = '/home/aistudio/data/data1272/word_dict.txt'
# TAG_DICT = '/home/aistudio/data/data1272/tag_dict.txt'
# """
# BATCH_SIZE = 128
# load Google vectors
# print("Loading word vector file GoogleNews-vectors-negative300.bin.gz......")
# print(time.asctime( time.localtime(time.time()) ))
# model = KeyedVectors.load_word2vec_format('/home/xiaoya/data/GoogleNews-vectors-negative300.bin.gz', binary=True)
# print(time.asctime( time.localtime(time.time()) ))


# =============================================get data from the dataset==============================================
def get_data(path):  # , train_valid_size
	"""
	extracting data for json file
	"""
	dm = DataManager(path, 'test')
	wv = dm.vector
	print(wv.shape)
	np.savetxt(path+"wv.txt", wv)
	train_data, test_data, dev_data = dm.data['train'], dm.data['test'], dm.data['dev']
	random.shuffle(train_data)
	w2id = dm.words
	relation2id = dm.relations
	tags_dict = {}
	x_train = []
	y_train = []
	for data in train_data:
		x_data, y_data = data_decoding(data, tags_dict)
		# appending each single data into the x_train/y_train sets
		x_train += x_data
		y_train += y_data

	x_test = []
	y_test = []
	for data in test_data:
		x_data, y_data = data_decoding(data, tags_dict)
		x_test += x_data
		y_test += y_data

	x_dev = []
	y_dev = []
	for data in dev_data:
		x_data, y_data = data_decoding(data, tags_dict)
		x_dev += x_data
		y_dev += y_data
	print(tags_dict)
	return w2id, relation2id, tags_dict, x_train, y_train, x_test, y_test, x_dev, y_dev


def data_decoding(data, tags_dict):
	"""
	decode the json file
	sentText is the sentence
	each sentence may have multiple types of relations
	for every single data, it contains: (sentence-splited, labels)
	"""
	sentence = data["sentext"]
	sentence_id = data["text"]
	# sentence = re.sub("[{}]+".format(string.punctuation), "", sentence)
	relations = data["relations"]
	# entities = data["entityMentions"]
	# entity_type = {}
	# entity_abbr = {"PERSON": "P", "LOCATION": "L", "ORGANIZATION": "G"}
	# for entity in entities:
	# 	en_name_words = entity["text"].split(" ")
	# 	en_type = entity["label"]
	# 	for en_name in en_name_words:
	# 		entity_type[en_name] = entity_abbr[en_type]
	x_data = []
	y_data = []

	y_data_id = []
	for i in relations:
		entity_1 = i["em1"].split(" ")
		entity_2 = i["em2"].split(" ")
		relation = i["type"]
		entity_tags = i['tags']
		entity_label_1 = entity_label_construction(entity_1)
		entity_label_2 = entity_label_construction(entity_2)
		output_list = sentence_label_construction(sentence, entity_label_1, entity_label_2, relation, entity_tags)
		# print(output_list)
		for each in output_list:
			if each not in tags_dict.keys():
				tags_dict[each] = len(tags_dict)
		x_data.append(sentence_id)  # (sentence.split(" "))
		y_data.append(output_list)
	for tags in y_data:
		tag_id = []
		for each in tags:
			tag_id.append(tags_dict[each])
		y_data_id.append(tag_id)

	return x_data, y_data_id


def entity_label_construction(entity):
	"""
	assign the label for each word in an entity.
	for entity with multiple words, it should follow the BIES rule
	"""
	relation_label = {}
	for i in range(len(entity)):
		if i == 0 and len(entity) >= 1:
			relation_label[entity[i]] = "B"
		if i != 0 and len(entity) >= 1 and i != len(entity) -1:
			relation_label[entity[i]] = "I"
		if i == len(entity) - 1 and len(entity) >= 1:
			relation_label[entity[i]] = "E"
		if i == 0 and len(entity) == 1:
			relation_label[entity[i]] = "S"
	return relation_label


def sentence_label_construction(sentence, relation_label_1, relation_label_2, relation, entity_tags):  # entity_type
	"""
	combine the label for each word in each entity with the relation
	and then combine the relation-entity label with the position of the entity in the triplet
	"""
	element_list = sentence.split()
	# if element_list[0] == "":
	# 	element_list = element_list[1:]
	dlist_1 = list(relation_label_1)
	dlist_2 = list(relation_label_2)
	output_list = []
	# max_sim = {}  # id: max_sim_value
	for ind in range(len(element_list)):
		i = element_list[ind]  # if i != "":
		if (i in dlist_1) and (entity_tags[ind] in [1, 4]):  # source entity
			output_list.append("%s-%s-1" % (relation, relation_label_1[i]))  # + entity_type[i]
		elif (i in dlist_2) and (entity_tags[ind] in [2, 5]):  # target entity
			output_list.append("%s-%s-2" % (relation, relation_label_2[i]))  # + entity_type[i]
		else:
			output_list.append('O')
		# calculate vector similarity
		# if relation != "None":
		# 	cor_sim = calculate_similarity(relation, i)
		# 	if len(max_sim) == 0:
		# 		max_sim[ind] = cor_sim
		# 	else:
		# 		if cor_sim > list(max_sim.values())[0]:
		# 			max_sim[ind] = cor_sim
	# if len(max_sim) != 0:
	# 	relation_word_id = list(max_sim.keys())[0]
	# 	if relation != "None" and output_list[relation_word_id] == 'O':
	# 		output_list[relation_word_id] = 'S-' + relation
	return output_list


if __name__ == '__main__':
	ROOT_DIR = "E:\\newFolder\\data\\entity&relation_dataset\\NYT11\\"  # ""C:/(O_O)!/thesis/5-RE with LSTM/code/HRL-RE-use/data/NYT11_demo/"
	w2id, relation2id, tags_dict, x_train, y_train, x_test, y_test, x_dev, y_dev = get_data(ROOT_DIR)
	# save_data(x_train, x_test, y_train, y_test)
	print("Writing data......")
	with open(ROOT_DIR + 'BSL_Tagging/data_train.pkl', 'wb') as outp:
		pickle.dump(x_train, outp)
		pickle.dump(y_train, outp)
		pickle.dump(w2id, outp)
		pickle.dump(relation2id, outp)
		pickle.dump(tags_dict, outp)

	with open(ROOT_DIR + 'BSL_Tagging/data_test.pkl', 'wb') as outp:
		pickle.dump(x_test, outp)
		pickle.dump(y_test, outp)
	with open(ROOT_DIR + 'BSL_Tagging/data_dev.pkl', 'wb') as outp:
		pickle.dump(x_dev, outp)
		pickle.dump(y_dev, outp)
	# relation_name = "/location/administrative_division/country"

	# text = "adbs., "
	# print(re.sub("[{}]+".format(string.punctuation), "", text))
