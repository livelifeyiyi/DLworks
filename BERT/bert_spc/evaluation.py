import json
import pandas as pd

json_file = "D:/remote/seg_predictions_0704.json" # prediction_0.597_self"  # prediction_souhu_0.597_onnews"  # prediction_souhu_testall"
with open(json_file, encoding='utf-8', mode='r') as infile:
	results = json.load(infile)

pred_labels = results['pred_labels']
label_ids = results['label_ids']
wrong_lines = []
test_file = "D:/remote/ground_truth2_seg.test"  # souhu_entity_sentiemnt_filter_2lines.title"  # title_emotions2lines"  # ground_truth_2lines_all.title"

# if linenum in wrong_lines:
# 		# 	print(str)
# act: {pred}


def print_wrong_labelled():
	data = {}
	with open(test_file, encoding='utf-8', mode='r') as infile:
		for linenum, string in enumerate(infile.readlines()):
			data[linenum] = string
	totalres = {'0': {'0': 0, '1': 0, '-1': 0, 'num': 0}, '1': {'0': 0, '1': 0, '-1': 0, 'num': 0}, '-1': {'0': 0, '1': 0, '-1': 0, 'num': 0}}
	for i in range(len(pred_labels)):
		if pred_labels[i] != label_ids[i]:
			# print(i+1, 2*i+1)
			print(data[2*i], "pred: %s, act: %s" % (pred_labels[i]-1, label_ids[i]-1))

			totalres[str(label_ids[i]-1)]['num'] += 1
			totalres[str(label_ids[i] - 1)][str(pred_labels[i]-1)] += 1
			# print("pred: %s, act: %s" % (pred_labels[i], label_ids[i]))
			# wrong_lines.append(2*i)
	print(totalres)


def json2pd():
	file_json= {}
	doc_id = 0
	with open("D:\\projects\\公司名识别\\ground_truth2.json", encoding='utf-8', mode='r') as infile:
		data = json.load(infile)
		for line_json in data:
			title = line_json["title"]  # title 作为一个句子
			content = line_json["content"]
			print(title)
	# 		file_json[doc_id] = {}
	# 		file_json[doc_id]['title'] = title
	# 		file_json[doc_id]['content'] = content
	# 		doc_id += 1
	# df = pd.DataFrame.from_dict(file_json)
	# df = df.transpose()
	# df.to_csv("file_json.csv")


def predict_vote():
	# with open(self.opt.output_name, encoding='utf-8', mode='r') as infile:
	# 	results = json.load(infile)
	# pred_labels = results['pred_labels']
	# label_ids = results['label_ids']
	# wrong_lines = []
	# test_file = self.opt.dataset_file['test']

	act_pred_label = {}  # id:{'entity': '', 'emotion': 0/1/-1, 'predictions': []}
	prev_entity = ""
	# "pred: %s, act: %s" % (pred_labels[i] - 1, label_ids[i] - 1)
	with open(test_file, encoding='utf-8', mode='r') as infile:
		lines = infile.readlines()
		pred_id = 0
		doc_id = 0
		for i in range(0, len(lines), 3):
			pred_id = int(i/3)
			doc = lines[i]
			entity = lines[i + 1].lower().strip()
			polarity_str = lines[i + 2].strip()
			assert int(polarity_str) == (label_ids[pred_id] - 1)
			# print(polarity_str, label_ids[pred_id] - 1)
			if prev_entity == "" or entity != prev_entity:
				doc_id += 1
				prev_entity = entity
				act_pred_label[doc_id] = {}
				act_pred_label[doc_id]['entity'] = entity
				act_pred_label[doc_id]['emotion'] = int(polarity_str)
				act_pred_label[doc_id]['predictions'] = [int(pred_labels[pred_id] - 1)]
			else:
				# if entity == prev_entity:
				act_pred_label[doc_id]['predictions'].append(int(pred_labels[pred_id] - 1))

	# print(act_pred_label)
	acc = 0.
	total = len(act_pred_label)
	for idx in act_pred_label.keys():
		each_pred = act_pred_label[idx]
		predic_labels = each_pred['predictions']
		act_label = each_pred['emotion']
		num = []
		num.append(predic_labels.count(-1))
		num.append(predic_labels.count(0))
		num.append(predic_labels.count(1))
		pred = num.index(max(num))
		if pred-1 == act_label:
			acc += 1
		else:
			act_pred_label[idx]['wrong'] = 'wrong'
			print(idx)
		act_pred_label[idx]['predict_count'] = num
		act_pred_label[idx]['predict_label'] = pred-1
	df = pd.DataFrame.from_dict(act_pred_label)  #.read_json(act_pred_label)
	df = df.transpose()
	df.to_csv("predict_results.csv")
	print(acc/total)


predict_vote()
# json2pd()
