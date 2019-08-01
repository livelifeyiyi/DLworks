import argparse
import json
import os
import re
from typing import List
import pandas as pd
import jieba
import torch
from pytorch_pretrained_bert import BertTokenizer
# import logging
#
# logger = logging.getLogger()


class BertData():
	def __init__(self, args):
		self.args = args
		self.tokenizer = BertTokenizer.from_pretrained(args.model_path, do_basic_tokenize=args.do_basic_tokenize)
		self.sep_vid = self.tokenizer.vocab['[SEP]']
		self.cls_vid = self.tokenizer.vocab['[CLS]']
		self.pad_vid = self.tokenizer.vocab['[PAD]']

	def preprocess(self, src: List[str], label: int):
		if len(src) == 0:
			return None
		src[-1] += '。'
		# original_src_txt = ['。'.join(src)]

		# labels = [0] * len(src)
		# for l in oracle_ids:
		# 	labels[l] = 1

		# idxs = [i for i, s in enumerate(src)]  # if (len(s) > self.args.min_src_ntokens)

		# src = [src[i][:self.args.max_src_ntokens] for i in idxs]
		# labels = [labels[i] for i in idxs]
		# src = src[:self.args.max_nsents]
		# labels = labels[:self.args.max_nsents]

		# if len(src) < self.args.min_nsents:
		# 	return None
		# if len(labels) == 0:
		# 	return None
		# src_txt = [' '.join(sent) for sent in src]
		# text = [' '.join(ex['src_txt'][i].split()[:self.args.max_src_ntokens]) for i in idxs]
		# text = [_clean(t) for t in text]
		text = '。 [SEP] [CLS] '.join(src)
		src_subtokens = self.tokenizer.tokenize(text)
		if len(src_subtokens) > args.max_src_ntokens - 2:
			src_subtokens = src_subtokens[:(args.max_src_ntokens-2)]
		src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']

		src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
		_segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
		segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
		segments_ids = []
		for i, s in enumerate(segs):
			if i % 2 == 0:
				segments_ids += s * [0]
			else:
				segments_ids += s * [1]
		cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
		# labels = labels[:len(cls_ids)]

		# tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
		# src_txt = [original_src_txt[i] for i in idxs]
		return src_subtoken_idxs, label, segments_ids, cls_ids, src


class Segments:
	def get_segments(self, sentences, args, entity_name, title):
		"""get segments by appending sentences"""
		max_char = args.max_src_ntokens - len(entity_name) - 2  # - sentence_num * 2
		sentences = [title + '。'] + sentences
		segments = []
		segment = ''
		i = 0
		cur_sent_num = 0
		while i <= len(sentences):
			if i == len(sentences):
				if len(segment) >= args.min_src_ntokens:
					segments.append(entity_name + '。' + segment)
				i += 1
				break
			if sentences[i] == '': continue
			eachsent = sentences[i] + '。'
			# eachsent = eachsent
			eachsent = eachsent.replace(' ', '')
			eachsent = eachsent.replace('\n', '')
			eachsent = eachsent.replace('\t', '')
			if len(segment) + len(eachsent) <= max_char - cur_sent_num * 2:  # each sent has [sep],[cls]
				segment += eachsent
				i += 1
				cur_sent_num += 1
			else:
				segments.append(entity_name + '。' + segment)
				cur_sent_num = 0
				segment = eachsent
				i += 1
		return segments

	def get_segments_by_entity(self, sentences, args, entity_name, title):
		"""get segments by the location of entities"""
		max_char = args.max_src_ntokens - len(entity_name) - 2
		segments = []
		# segment = ''
		seg_char = '。'
		all_sentencs = seg_char.join(sentences)
		all_sentencs = title + seg_char + all_sentencs
		all_sentencs = all_sentencs.replace(' ', '')
		all_sentencs = all_sentencs.replace('\n', '')
		all_sentencs = all_sentencs.replace('\t', '')
		if len(all_sentencs) <= max_char:
			return [entity_name + seg_char + title + seg_char + all_sentencs]
		entity_start_id =[]
		for i, char in enumerate(all_sentencs):
			if char == entity_name[0]:
				if all_sentencs[i:i+len(entity_name)] == entity_name:
					entity_start_id.append(i)
			continue
		for id in entity_start_id:
			l_id = id - (max_char-len(entity_name)) // 2
			r_id = id + (max_char-len(entity_name)) // 2
			if l_id < 0:
				segment = all_sentencs[0:max_char]
			elif r_id > len(all_sentencs):
				segment = all_sentencs[len(all_sentencs)-max_char:-1]
			else:
				segment = all_sentencs[l_id:r_id+1]
			sentence_num = segment.count(seg_char) + 2

			segment = segment[sentence_num:(len(segment)-sentence_num)]
			segment = entity_name + seg_char + segment
			segments.append(segment)

		return segments


def process_lie_segment(line_json, datasets, bert):
	Seg = Segments()
	emotion_dict = {'NORM': 1, 'NEG': 0, 'POS': 2}  # 0,-1,1
	# emotion_dict = {"正向": 1, "负向": -1, "中性": 0}  #
	title = line_json["title"]  # title 作为一个句子
	content = line_json["content"].replace('\n', '').replace('\r', '')
	for entity in line_json["coreEntityEmotions"]:
		entity_name = entity["entity"]
		entity_emotion = entity["emotion"]
		# process content, each sentence
		sentences = content.split('。')  # re.split('([。])', content)
		# sentences = [entity_name] + [title] + sentences  # +'。'
		# sentence_num = len(sentences)
		segments = Seg.get_segments(sentences, args, entity_name, title)
		# segments = Seg.get_segments_by_entity(sentences, args, entity_name, title)
		# jieba.add_word(entity_name)

		for segment in segments:
			# if entity_name in segment:
			# res += segment + '\n' + entity_name + '\n' + str(emotion_dict[entity_emotion]) + '\n'
			# segemnt_cut = jieba.cut(segment)
			# content_cut_list = [word for word in segemnt_cut if word and word != ' ' and word != '\n' and word!='\t']
			# content_str = ' '.join(content_cut_list)

			b_data = bert.preprocess([i for i in segment.split('。') if i], emotion_dict[entity_emotion])
			if b_data is None:
				print(line_json["newsId"])
				continue
			indexed_tokens, labels, segments_ids, cls_ids, src_txt = b_data
			b_data_dict = {"src": indexed_tokens, "labels": labels, "segs": segments_ids, 'clss': cls_ids,
			               'src_txt': src_txt}
			datasets.append(b_data_dict)

	return datasets


def process_lie_segment_test(line_json, datasets, bert):
	Seg = Segments()
	emotion_dict = {"正向": 2, "负向": 0, "中性": 1}  # 1, -1, 0
	title = line_json["title"]  # title 作为一个句子
	content = line_json["content"].replace('\n', '').replace('\r', '')
	# for news.xlsx
	entities = str(line_json["corporations"]).split('、')
	emotions = line_json['emotion'].split('、')
	for i in range(len(entities)):
		entity_name = entities[i]
		entity_emotion = emotions[i]
		# process content, each sentence
		sentences = content.split('。')  # re.split('([。])', content)
		sentences = [entity_name] + [title] + sentences  # +'。'
		# sentence_num = len(sentences)
		# max_char = args.max_src_ntokens - sentence_num * 2
		segments = Seg.get_segments(sentences, args, entity_name, title)

		jieba.add_word(entity_name)

		for segment in segments:
			# if entity_name in segment:
			# res += segment + '\n' + entity_name + '\n' + str(emotion_dict[entity_emotion]) + '\n'
			# segemnt_cut = jieba.cut(segment)
			# content_cut_list = [word for word in segemnt_cut if word and word != ' ' and word != '\n' and word!='\t']
			# content_str = ' '.join(content_cut_list)

			b_data = bert.preprocess([i for i in segment.split('。') if i], emotion_dict[entity_emotion])
			if b_data is None:
				print(line_json["corporations"])
				continue
			indexed_tokens, labels, segments_ids, cls_ids, src_txt = b_data
			b_data_dict = {"src": indexed_tokens, "labels": labels, "segs": segments_ids, 'clss': cls_ids,
			               'src_txt': src_txt}
			datasets.append(b_data_dict)

	return datasets


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# parser.add_argument("-mode", default='../bert-base-chinese', type=str, help='format_to_lines or format_to_bert')
	# parser.add_argument("-map_path", default='../data/')
	parser.add_argument("--input_file", default='../data/demo_data.json')
	parser.add_argument("--save_path", default='../data/')
	parser.add_argument("--model_path", default='../bert-base-chinese', type=str, help="The pre-trained model path")
	parser.add_argument('--do_basic_tokenize', default=True, action='store_true')
	parser.add_argument('--test_data', default=False, help="whether the input is a xlsx test data")
	# parser.add_argument("-shard_size", default=2000, type=int)
	# parser.add_argument('-min_nsents', default=3, type=int)
	# parser.add_argument('-max_nsents', default=100, type=int)
	parser.add_argument('--min_src_ntokens', default=30, type=int)  # drop the segments which are shorter than min_src_ntokens
	parser.add_argument('--max_src_ntokens', default=512, type=int)

	# parser.add_argument("-lower", type=str2bool, nargs='?',const=True,default=True)

	parser.add_argument('-log_file', default='../../logs/cnndm.log')

	# parser.add_argument('-dataset', default='', help='train, valid or test, defaul will process all datasets')

	# parser.add_argument('-n_cpus', default=2, type=int)

	args = parser.parse_args()
	bert = BertData(args)
	datasets = []
	# ROOT = "../data/"
	# json_file = ROOT + "demo_data.json"
	# target_file = ROOT + "demo_data_bertsum.train"
	if not args.test_data:
		with open(args.input_file, encoding='utf-8', mode='r') as infile:
			entity_count = {}
			sentences_num = []
			data = json.load(infile)
			count = 0
			for each in data:
				datasets = process_lie_segment(each, datasets, bert)
				count += 1
				if count == len(data) // 10:
					print(count)
					print('Saving validation set to %s' % args.save_path)
					torch.save(datasets, args.save_path + 'valid.data')
					datasets = []

			print('Saving training data to %s' % args.save_path)
			torch.save(datasets, args.save_path + 'train.data')
			total_num = len(datasets)
			print('Number of document: %s, Number of data :%s' % (count, total_num))

	if args.test_data:
		df = pd.read_excel(args.input_file)
		for i in range(df.shape[0]):
			datasets = process_lie_segment_test(df.loc[i], datasets, bert)
		print('Saving test data to %s' % args.save_path)
		torch.save(datasets, args.save_path + 'test.data')
		total_num = len(datasets)
		print('Number of document: %s, Number of data :%s' % (df.shape[1], total_num))
