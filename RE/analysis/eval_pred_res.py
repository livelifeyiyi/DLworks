import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


def eval_file(json_file, prf=True, pr_curve=False):
	with open(json_file, encoding='utf-8', mode='r') as infile:
		results = json.load(infile)
	RE_predict = results["RE_predict"]
	RE_actual = results["RE_actual"]
	RE_output_logits = results["RE_output_logits"]
	NER_predict = results["NER_predict"]
	NER_actual = results["NER_actual"]
	NER_output_logits = results["NER_output_logits"]
	RE_actual_one = np.array(RE_actual).reshape(-1)
	RE_predict_one = np.array(RE_predict).reshape(-1)
	if prf:
		print(set(RE_predict_one), set(RE_actual_one))
		RE_res = metrics.classification_report(RE_actual_one, RE_predict_one)
		print('RE Prediction results: \n{}'.format(RE_res))
	if pr_curve:
		draw_pr_curve(RE_actual_one, RE_output_logits)

	NER_actual_one = []
	NER_predict_one = []
	for i in range(len(NER_actual)):
		if isinstance(NER_actual[i], list):
			NER_actual_batch = []
			NER_predict_batch = []
			for each_bacth_id in range(len(NER_actual[i])):
				mask = 1 - (np.array(NER_actual[i][each_bacth_id]) == 7)
				NER_actual_batch += NER_actual[i][each_bacth_id][:np.count_nonzero(mask)]
				NER_predict_batch += NER_predict[i][each_bacth_id][:np.count_nonzero(mask)]
			NER_actual_one += NER_actual_batch
			NER_predict_one += NER_predict_batch
		else:
			NER_actual_one += NER_actual[i]
			NER_predict_one += NER_predict[i]
	if prf:
		print(set(NER_predict_one), set(NER_actual_one))
		NER_res = metrics.classification_report(np.array(NER_actual_one).reshape(-1), np.array(NER_predict_one).reshape(-1))
		print('NER Prediction results: \n{}'.format(NER_res))


def draw_pr_curve(RE_actual_one, RE_output_logits):
	precision = dict()
	recall = dict()
	average_precision = dict()
	RE_output_logits_np = np.array(RE_output_logits).reshape(-1, 13)
	#
	#  for i in range(len(RE_tag)):
	# 	RE_binary_label = 0 + (RE_actual_one == i)
	# 	precision[i], recall[i], _ = precision_recall_curve(RE_binary_label,
	# 														RE_output_logits_np[:, i])
	# 	average_precision[i] = average_precision_score(RE_binary_label, RE_output_logits_np[:,])

	# A "micro-average": quantifying score on all classes jointly
	RE_actual_binarized = label_binarize(RE_actual_one, classes=RE_tag)
	auc_score = roc_auc_score(RE_actual_binarized, RE_output_logits_np, average="micro")
	print("AUC score: %.4f" % auc_score)
	precision["micro"], recall["micro"], _ = precision_recall_curve(RE_actual_binarized.ravel(),
																	RE_output_logits_np.ravel())
	average_precision["micro"] = average_precision_score(RE_actual_binarized, RE_output_logits_np,
														 average="micro")
	print('Average precision score, micro-averaged over all classes: {0:0.2f}'
		  .format(average_precision["micro"]))
	plt.figure()
	plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
			 where='post')
	plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b')

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title(
		'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
			.format(average_precision["micro"]))
	plt.show()


RE_tag = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
NER_tag = [0, 1, 2, 3, 4, 5, 6]
# json_file = "D:/code/RE/BERT_LSTM_LSTM/mode_pw/pred_res/predict_test_epoch_1.json"
# print("test dataset")
# eval_file(json_file, False, True)

# json_file = "D:/code/RE/BERT_LSTM_LSTM/model_dropout0.7/epoch2predict_dev.json"
json_file = "D:/code/RE/BERT_LSTM_LSTM/mode_pw/pred_res/predict_dev.json"
print("dev dataset")
eval_file(json_file, False, True)
