
import numpy as np
import torch
import torch.autograd as autograd


def calcF1(acc, cnt, tot, beta=1.0):
	if cnt == 0 or tot == 0:
		return 0
	precision = float(acc) / float(cnt)
	recall = float(acc) / float(tot)
	if recall > 1:
		recall = 1
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


def calcREReward(top_action, gold_labels):
	# if predicted action == elation_label, r=1, else r=0
	lenth = len(top_action)  # round number
	r = [0. for i in range(lenth)]
	# rem = [0 for i in range(len(gold_labels))]
	for i in range(lenth)[::-1]:
		if top_action[i] > 0:
			ok = -1
			if isinstance(gold_labels, int):
				if gold_labels == top_action[i]:
					ok = 1
			else:
				# for j, label in enumerate(gold_labels):
				if gold_labels[i] == top_action[i]:
					ok = 1
					# if rem[j] == 0:
					# 	ok = 0.5
					# 	rem[j] = 1
					# 	break
					# else:
					# 	ok = -0.2
			r[i] = ok
	return r


def calcREFinalReward(RE_actions, labels, top_bias=0.):
	r = 0.
	sample_round = len(RE_actions)
	acc, cnt, tot = 0, 0, 1
	if isinstance(labels, int):
		tot = 1
	elif isinstance(labels, list):
		tot = len(labels)

	for i in range(sample_round):
		if tot == 1:
			if RE_actions[i] == labels:  # ['type']:
				acc += 1
			if RE_actions[i] > 0:
				cnt += 1
		else:
			for label in labels:
				if RE_actions[i] == label:  # ['type']:
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


def calcEntityReward(bot_action, gold_labels):
	lenth = len(gold_labels)
	r = [0. for i in range(lenth)]  # for j in range(len(bot_action))]
	# j = 0
	# for i in range(lenth):
		# if top_action[i] > 0:
			# for label in gold_labels:
				# if label['type'] == top_action[i]:
	for t in range(lenth):
		if gold_labels[t] == bot_action[t]:
			if gold_labels[t] in [1, 2, 4, 5]:
				r[t] = 1  # source and target entities
			elif gold_labels[t] in [3, 6]:
				r[t] = 0.7  # non-concerned entities
			elif gold_labels[t] in [0]:
				r[t] = 0.3  # non-entities
		else:  # wrong labeled
			r[t] = -0.5
	# j += 1
	return r


def calcEntityFinalReward(bot_action, gold_labels, bot_bias=0.):
	lenth = len(gold_labels)
	r = [0. for j in range(lenth)]
	# j = 0
	for i in range(lenth):
		# if top_action[i] > 0:
		r[i] = -1.0
			# for label in gold_labels:
				# if label['type'] == top_action[i]:
		if gold_labels[i] == bot_action[i]:
			ok = True
		# for t in range(lenth):
		if gold_labels[i] != bot_action[i]:
			ok = False
			# break
		if ok:
			r[i] = 1.0

	for j in range(lenth):
		r[j] -= bot_bias
	return r


def calcBotGrad(bot_action, bot_actprob, bot_reward, bot_final_reward, pretrain=False):
	lenth = len(bot_reward)
	bot_tot_reward = [0. for i in range(lenth)]
	if torch.cuda.is_available():
		grads = autograd.Variable(torch.cuda.FloatTensor(1, ).fill_(0), requires_grad=True)
		# bot_actprob = torch.cuda.FloatTensor(bot_actprob)
	else:
		grads = autograd.Variable(torch.FloatTensor(1, ).fill_(0), requires_grad=True)
		# bot_actprob = torch.FloatTensor(bot_actprob)
	j = 0
	for i in range(lenth):
		# if top_action[i] > 0:
			bot_tot_reward[i] = bot_reward[j] / lenth + bot_final_reward[j]  #
			# for k in range(lenth)[::-1]:
			tmp = bot_actprob[j]
			if tmp < 0:
				tmp = 1 - tmp
			to_grad = -torch.log(tmp)  # [k]
			if not pretrain:
				if torch.cuda.is_available():
					to_grad = to_grad * autograd.Variable(torch.cuda.FloatTensor(1, ).fill_(bot_tot_reward[i]), requires_grad=True)
				else:
					to_grad = to_grad * autograd.Variable(torch.FloatTensor(1, ).fill_(bot_tot_reward[i]), requires_grad=True)
			if bot_action[j] == 0:  # [k]
				to_grad *= 0.3
			elif bot_action[j] == 3 or bot_action[j] == 6:  # [k]
				to_grad *= 0.7
			else:
				to_grad *= 1.0
			grads = grads + to_grad
			j += 1
	return bot_tot_reward, grads


def calcTopGrad(top_action, top_actprob, top_reward, top_final_reward, pretrain=False):
	lenth = len(top_action)
	decay_reward = top_final_reward
	if torch.cuda.is_available():
		grads = autograd.Variable(torch.cuda.FloatTensor(1, ).fill_(0), requires_grad=True)
		# top_actprob = torch.cuda.FloatTensor(top_actprob)
	else:
		grads = autograd.Variable(torch.FloatTensor(1, ).fill_(0), requires_grad=True)
		# top_actprob = torch.FloatTensor(top_actprob)
	for i in range(lenth)[::-1]:
		decay_reward = decay_reward * 0.95 + top_reward[i]
		to_grad = -torch.log(top_actprob[i]).float()
		if not pretrain:
			if torch.cuda.is_available():
				to_grad = to_grad * autograd.Variable(torch.cuda.FloatTensor(1, ).fill_(decay_reward), requires_grad=True)
			else:
				to_grad = to_grad * autograd.Variable(torch.FloatTensor(1, ).fill_(decay_reward), requires_grad=True)
		if top_action[i] == 0:
			to_grad *= 0.3
		grads = grads + to_grad
	return grads





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


def optimize(RE_actions, top_actprobs, relation_label, entity_label, entity_actions, entity_probs, entity_loss):  #
	sample_round = len(RE_actions)

	top_bias = 0.
	# for i in range(sample_round):
	top_bias += calcREFinalReward(RE_actions, relation_label, 0.)
	top_bias /= sample_round
	top_reward = calcREReward(RE_actions, relation_label)
	# grads = autograd.Variable(torch.cuda.FloatTensor(1, ).fill_(0))

	# bot_bias, bot_cnt = 0., 0
	tmp = calcEntityFinalReward(entity_actions, entity_label, 0.)
	# bot_cnt += len(tmp)
	# bot_bias += np.sum(tmp)
	bot_bias = -entity_loss

	bot_reward = calcEntityReward(entity_actions, entity_label)
	bot_final_reward = [x-bot_bias for x in tmp]
	# bot_final_reward = calcEntityFinalReward(entity_actions, entity_label, bot_bias)
	# tmp-= bot_bias
	bot_tot_reward, grads = calcBotGrad(entity_actions, entity_probs, bot_reward, bot_final_reward)
	for i in range(sample_round):
		top_reward[i] += bot_tot_reward[0]
	# if "RE" in mode:
	grads += calcTopGrad(RE_actions, top_actprobs, top_reward, top_bias, pretrain=False)
	loss = grads.cpu().data[0]
	# grads.backward(retain_graph=True)

	#
	# loss = .0
	# for i in range(sample_round):
	# 	loss += optimize(RE_actions[i], top_actprobs[i], labels, top_bias)
	return grads  # loss   # / sample_round

#
# RE_actions = [torch.from_numpy(np.array([11])), torch.from_numpy(np.array([0])), torch.from_numpy(np.array([11])),
# 				torch.from_numpy(np.array([3])), torch.from_numpy(np.array([5]))]
# top_actprobs = [torch.from_numpy(np.array([0.0757])), torch.from_numpy(np.array([0.0798])), torch.from_numpy(np.array([0.0763])),
# 				torch.from_numpy(np.array([0.0764])), torch.from_numpy(np.array([0.0743]))]
# relation_label = 4
# entity_label = [0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# entity_actions = torch.tensor([1, 3, 3, 3, 3, 3, 4, 4, 2, 2, 2, 2, 2, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 2,
#         2, 2, 6, 6, 2, 2, 2, 2, 2, 6, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 4, 2, 2,
#         0, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 5, 5, 5, 0, 0, 0, 0, 4, 4])
# entity_probs = [torch.tensor(-4.2052), torch.tensor(-4.2133), torch.tensor(-4.2154), torch.tensor(-4.2171), torch.tensor(-4.2183), torch.tensor(-4.2192), torch.tensor(-4.2194), torch.tensor(-4.2194), torch.tensor(-4.2189), torch.tensor(-4.2187), torch.tensor(-4.2186), torch.tensor(-4.2186), torch.tensor(-4.2187), torch.tensor(-4.2187), torch.tensor(-4.2186), torch.tensor(-4.2187), torch.tensor(-4.2187), torch.tensor(-4.2188), torch.tensor(-4.2187), torch.tensor(-4.2186), torch.tensor(-4.2186), torch.tensor(-4.2188), torch.tensor(-4.2189), torch.tensor(-4.2191), torch.tensor(-4.2190), torch.tensor(-4.2190), torch.tensor(-4.2188), torch.tensor(-4.2185), torch.tensor(-4.2184), torch.tensor(-4.2182), torch.tensor(-4.2182), torch.tensor(-4.2182), torch.tensor(-4.2183), torch.tensor(-4.2182), torch.tensor(-4.2178), torch.tensor(-4.2177), torch.tensor(-4.2176), torch.tensor(-4.2176), torch.tensor(-4.2179), torch.tensor(-4.2183), torch.tensor(-4.2185), torch.tensor(-4.2185), torch.tensor(-4.2184), torch.tensor(-4.2185), torch.tensor(-4.2188), torch.tensor(-4.2190), torch.tensor(-4.2191), torch.tensor(-4.2192), torch.tensor(-4.2193), torch.tensor(-4.2192), torch.tensor(-4.2189), torch.tensor(-4.2187), torch.tensor(-4.2187), torch.tensor(-4.2188), torch.tensor(-4.2187), torch.tensor(-4.2185), torch.tensor(-4.2183), torch.tensor(-4.2183), torch.tensor(-4.2184), torch.tensor(-4.2185), torch.tensor(-4.2184), torch.tensor(-4.2183), torch.tensor(-4.2183), torch.tensor(-4.2180), torch.tensor(-4.2177), torch.tensor(-4.2175), torch.tensor(-4.2160), torch.tensor(-4.2135)]
#
# optimize(RE_actions, top_actprobs, relation_label, entity_label, entity_actions, entity_probs, torch.tensor(-1.2133))
