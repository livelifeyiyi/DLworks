
import numpy as np
import torch
import torch.autograd as autograd


def calcF1(acc, cnt, tot, beta=1.0):
	if cnt == 0 or tot == 0:
		return 0
	precision = float(acc) / float(cnt)
	recall = float(acc) / float(tot)
	if precision + recall < 1e-5:
		return 0
	return (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)

"""
def calc_re_acc(RE_action, bot_action, gold_labels):
	if RE_action == gold_labels:
		acc = 1
	else:
		acc = 0
	acc, cnt, tot = 0, 0, len(gold_labels)
	used = [0 for i in range(len(top_action))]
	for label in gold_labels:
		tp, tags = label['type'], label['tags']
		j, ok = 0, 0
		for i in range(len(top_action)):
			if top_action[i] == tp and tp > 0 and used[i] == 0 and ok == 0:
				match = 1
				if "NER" in mode:
					for k in range(len(bot_action[j])):
						if tags[k] == 4 and bot_action[j][k] != 4:
							match = 0
						if tags[k] != 4 and bot_action[j][k] == 4:
							match = 0
						if tags[k] == 5 and bot_action[j][k] != 5:
							match = 0
						if tags[k] != 5 and bot_action[j][k] == 5:
							match = 0
				if match == 1:
					ok = 1
					used[i] = 1
			if top_action[i] > 0:
				j += 1
				cnt += 1
		acc += ok
	cnt //= tot
	return acc, tot, cnt
"""


def calcREFinalReward(RE_actions, labels, top_bias=0.):
	r = 0.
	sample_round = len(RE_actions)
	acc, cnt, tot = 0, 0, len(labels)

	for i in range(sample_round):
		for label in labels:
			if RE_actions[i] == label['type']:
				acc += 1
			if RE_actions[i] > 0:
				cnt += 1
	# a1, t1, c1 = calc_re_acc(top_action, None, gold_labels)
	if cnt != 0:
		r = calcF1(acc, cnt, tot, beta=0.9)
	else:
		r = -2
	if cnt > tot:
		r -= 0.5 * (cnt - tot)
	# r *= len(RE_actions)   # why?????
	return r - top_bias


def calcBotReward(bot_action, gold_labels):
	lenth = len(bot_action)
	r = [[0. for i in range(lenth)] for j in range(len(bot_action))]
	j = 0
	for i in range(lenth):
		# if top_action[i] > 0:
			for label in gold_labels:
				# if label['type'] == top_action[i]:
					for t in range(lenth):
						if label['tags'][t] == bot_action[j][t]:
							if label['tags'][t] in [4, 5, 6]:
								r[j][t] = 0.5
							elif label['tags'][t] in [1, 2, 3]:
								r[j][t] = 0.2
						else:
							r[j][t] = -0.5
			j += 1
	return r


def calcBotFinalReward(bot_action, gold_labels, bot_bias=0.):
	lenth = len(gold_labels)
	r = [0. for j in range(len(bot_action))]
	j = 0
	for i in range(lenth):
		# if top_action[i] > 0:
			r[j] = -1.0
			for label in gold_labels:
				# if label['type'] == top_action[i]:
					ok = True
					for t in range(lenth):
						if label['tags'][t] != bot_action[j][t]:
							ok = False
							break
					if ok:
						r[j] = 1.0
			j += 1
	for j in range(len(bot_action)):
		r[j] -= bot_bias
	return r


def calcBotGrad(bot_action, bot_actprob, bot_reward, bot_final_reward, pretrain=False):
	lenth = len(bot_action)
	bot_tot_reward = [0. for i in range(lenth)]
	grads = autograd.Variable(torch.cuda.FloatTensor(1, ).fill_(0))
	j = 0
	for i in range(lenth):
		# if top_action[i] > 0:
			bot_tot_reward[i] = sum(bot_reward[j]) / lenth + bot_final_reward[j]  #
			for k in range(lenth)[::-1]:
				to_grad = -torch.log(bot_actprob[j][k])
				if not pretrain:
					to_grad *= autograd.Variable(torch.cuda.FloatTensor(1, ).fill_(bot_tot_reward[i]))
				if bot_action[j][k] == 0:
					to_grad *= 0.3
				elif bot_action[j][k] == 3 or bot_action[j][k] == 6:
					to_grad *= 0.7
				else:
					to_grad *= 1.0
				grads = grads + to_grad
			j += 1
	return bot_tot_reward, grads


def calcTopGrad(top_action, top_actprob, top_reward, top_final_reward, pretrain=False):
	lenth = len(top_action)
	decay_reward = top_final_reward
	grads = autograd.Variable(torch.cuda.FloatTensor(1, ).fill_(0))
	for i in range(lenth)[::-1]:
		decay_reward = decay_reward * 0.95 + top_reward[i]
		to_grad = -torch.log(top_actprob[i])
		if not pretrain:
			to_grad *= autograd.Variable(torch.cuda.FloatTensor(1, ).fill_(decay_reward))
		if top_action[i] == 0:
			to_grad *= 0.3
		grads = grads + to_grad
	return grads


def calcTopReward(top_action, gold_labels):
	lenth = len(top_action)
	r = [0. for i in range(lenth)]
	rem = [0 for i in range(len(gold_labels))]
	for i in range(lenth)[::-1]:
		if top_action[i] > 0:
			ok = -1
			for j, label in enumerate(gold_labels):
				if label['type'] == top_action[i]:
					if rem[j] == 0:
						ok = 0.5
						rem[j] = 1
						break
					else:
						ok = -0.2
			r[i] = ok
	return r




'''
def optimize(top_action, top_actprob, gold_labels, top_bias=0.):
	# lenth = len(top_action)
	top_reward = calcTopReward(top_action, gold_labels)
	top_final_reward = calcREFinalReward(top_action, gold_labels, top_bias)
	# pretrain = True if "pretrain" in mode else False
	pretrain = False
	# if "NER" in mode:
	# 	bot_reward = calcBotReward(top_action, bot_action, gold_labels)
	# 	bot_final_reward = calcBotFinalReward(top_action, bot_action, gold_labels, bot_bias)
	# 	bot_tot_reward, grads = calcBotGrad(top_action, bot_action, bot_actprob, bot_reward, bot_final_reward, pretrain)
	# 	for i in range(lenth):
	# 		top_reward[i] += bot_tot_reward[i]
	# else:
	grads = autograd.Variable(torch.cuda.FloatTensor(1, ).fill_(0))
	# if "RE" in mode:
	grads += calcTopGrad(top_action, top_actprob, top_reward, top_final_reward, pretrain)
	loss = grads.cpu().data[0]
	grads.backward()
	return loss
'''


def optimize(RE_actions, top_actprobs, labels, entity_actions, entity_probs):
	sample_round = len(RE_actions)

	top_bias = 0.
	# for i in range(sample_round):
	top_bias += calcREFinalReward(RE_actions, labels, 0.)
	top_bias /= sample_round
	top_reward = calcTopReward(RE_actions, labels)
	# grads = autograd.Variable(torch.cuda.FloatTensor(1, ).fill_(0))

	bot_bias, bot_cnt = 0., 0
	tmp = calcBotFinalReward(entity_actions, labels, 0.)
	bot_cnt += len(tmp)
	bot_bias += np.sum(tmp)

	bot_reward = calcBotReward(entity_actions, labels)
	bot_final_reward = calcBotFinalReward(entity_actions, labels, bot_bias)
	bot_tot_reward, grads = calcBotGrad(entity_actions, entity_probs, bot_reward, bot_final_reward)
	for i in range(sample_round):
		top_reward[i] += bot_tot_reward[0]
	# if "RE" in mode:
	grads += calcTopGrad(RE_actions, top_actprobs, top_reward, top_bias, pretrain=False)
	loss = grads.cpu().data[0]
	grads.backward()
	#
	# loss = .0
	# for i in range(sample_round):
	# 	loss += optimize(RE_actions[i], top_actprobs[i], labels, top_bias)
	return loss   # / sample_round
