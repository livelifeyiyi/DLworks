from sklearn import metrics
from bertTransformer.others.logging import logger


def predict_vote(pred_labels, label_ids, test_dataloader):

		act_pred_label = {}  # id:{'entity': '', 'emotion': 0/1/-1, 'predictions': []}
		prev_entity = ""
		# "pred: %s, act: %s" % (pred_labels[i] - 1, label_ids[i] - 1)

		doc_id = 0
		for i, test_dataset in enumerate(test_dataloader):
			# pred_id = int(i / 3)
			doc = test_dataset['src_txt']
			entity = doc[0]
			polarity = test_dataset['labels']
			assert int(polarity) == (label_ids[i])
			# print(polarity_str, label_ids[pred_id] - 1)
			if prev_entity == "" or entity != prev_entity:
				doc_id += 1
				prev_entity = entity
				act_pred_label[doc_id] = {}
				act_pred_label[doc_id]['entity'] = entity
				act_pred_label[doc_id]['emotion'] = int(polarity)  # actual label;
				act_pred_label[doc_id]['predictions'] = [int(pred_labels[i])]  # predict label
			else:
				# if entity == prev_entity:
				act_pred_label[doc_id]['predictions'].append(int(pred_labels[i]))

		# print(act_pred_label)
		acc = 0.
		total = len(act_pred_label)
		act_labels_all = []
		pred_labels_all = []
		for idx in act_pred_label.keys():
			each_pred = act_pred_label[idx]
			predic_labels = each_pred['predictions']
			act_label = each_pred['emotion']
			num = []
			num.append(predic_labels.count(0))
			num.append(predic_labels.count(1))
			num.append(predic_labels.count(2))
			# 0:0, 1:1, 2:2
			if num[0] == num[1] and num[0] != 0 and num[0] > num[2]:
				pred = 0
			elif num[1] == num[2] and num[1] != 0 and num[1] > num[0]:
				pred = 2
			elif num[0] == num[2] and num[0] != 0 and num[0] > num[1]:
				pred = 1
			else:
				pred = num.index(max(num))
			# if num[0]== num[1] or num[1] == num[2] or num[0] == num[2]:
			# 	print(pred, num)
			if pred == act_label:
				acc += 1
			act_labels_all.append(act_label)
			pred_labels_all.append(pred)
		# print(acc / total)
		# vote_f1 = metrics.f1_score(act_labels_all, pred_labels_all, labels=[0, 1, 2], average=None)
		# vote_recall = metrics.recall_score(act_labels_all, pred_labels_all, labels=[0, 1, 2], average=None)
		# logger.info('>> vote_acc: {:.4f}, vote_recall:{} vote_f1: {}'.format(acc / total, vote_recall, vote_f1))
		pred_result = metrics.classification_report(act_labels_all, pred_labels_all, target_names=['NEG', 'NEU', "POS"])
		logger.info('>> vote_acc: {:.4f}'.format(acc / total))
		logger.info('>> Prediction voted results: \n {}'.format(pred_result))
