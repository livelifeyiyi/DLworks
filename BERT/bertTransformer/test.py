import torch
from torch.utils.data import DataLoader
dataset1 = torch.load('../data/train_ht.data')
# dataset2 = torch.load('../data/train_char.data')

dataloader1 = DataLoader(dataset=dataset1, batch_size=3, shuffle=True)
for i, batch in enumerate(dataloader1):
	src, labels, segs = batch['src'], batch['labels'], batch['segs']
	pass
pass
# a = 0.7928
# b = 0.8136
# print((b-a)/a)