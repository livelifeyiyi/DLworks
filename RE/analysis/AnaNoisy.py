import codecs
import ast
import numpy as np
import pandas as pd

ROOT = "E:/newFolder/code/RE/RE_RL_output/20190513_NYT11/"


def doc2csv():
	filename = ROOT + "test_noisy.txt"  # "test_noisy_for_ana.txt"
	mycolumns = ["noisy_num", "sim_prob", "relation_name", "source_entity", "target_entity", "sentence"]
	rounds_num = 0.
	df = pd.DataFrame(columns=mycolumns)
	with codecs.open(filename, mode='r', encoding="utf-8") as infile:
		line = infile.readline()
		line_num = 0
		while line:
			line_num += 1
			# if line_num < 1763200-128000:
			# 	line = infile.readline()
			# 	continue
			# row = []
			eles = line.strip("\n").strip("\r").split(",\t")
			noisy_rounds, sim_prob, relation_name, entity_tag, sentence = eles[0], eles[1], eles[2], eles[3], eles[4]
			noisy_rounds = ast.literal_eval(noisy_rounds)
			# print(noisy_rounds[-1])

			rounds_num += len(noisy_rounds)
			noisy_num = np.sum(np.array(noisy_rounds))
			entity_tag = ast.literal_eval(entity_tag)
			sentence_words = sentence.split(" ")
			source_idx, target_idx = [], []
			for idx, tag in enumerate(entity_tag):
				if tag == 4:
					source_idx.append(idx)
				if tag == 1:
					source_idx.append(idx)
				if tag == 5:
					target_idx.append(idx)
				if tag == 2:
					target_idx.append(idx)
			source_entity, target_entity = "", ""
			for i in target_idx:
				if sentence_words[0] == "":
					source_entity += sentence_words[i+1]
				else:
					source_entity += sentence_words[i]
				source_entity += " "
			for j in source_idx:
				if sentence_words[0] == "":
					target_entity += sentence_words[j + 1]
				else:
					target_entity += sentence_words[j]
				target_entity += " "
			row = [noisy_num, float(sim_prob), relation_name, source_entity, target_entity, sentence]
			df.loc[len(df)] = row
			line = infile.readline()
		df.to_csv(ROOT+"test_noisy.csv", index=False)


def ana_csv():
	file_name = ROOT + "test_noisy_for_ana_nyt11.csv"
	df = pd.read_csv(file_name, index_col=None, header=0)
	round_num, noisy_num, sim_prob, tag = df['round_num'], df['noisy_num'], df['sim_prob'], df['tag_by_human']
	rl_last = df['round_last']
	rl_acc = 0.
	rl_last_acc = 0.
	sim_acc = 0.
	core = {0: 1, 1: 0}
	for idx, tag_value in enumerate(tag):
		RL_percent = float(noisy_num[idx])/round_num[idx]
		sim = sim_prob[idx]
		if core[rl_last[idx]] == tag_value:
			rl_last_acc += 1
		if ((1-RL_percent) >= 0.5 and tag_value == 1) or ((1-RL_percent) < 0.5 and tag_value == 0):
			rl_acc += 1
		if (sim >= 0.23 and tag_value == 1) or (sim < 0.23 and tag_value == 0):
			sim_acc += 1
	print(rl_acc, sim_acc, rl_last_acc)
	print(rl_acc/len(tag), sim_acc/len(tag), rl_last_acc/len(tag))


ana_csv()
# doc2csv()
# print(rounds_num, line_num)
# print(rounds_num/1000)
