import torch
# from argparse import ArgumentParser

# parser = ArgumentParser(add_help=False)
# parser.add_argument(
#     "--gpu", default="1",
#     type=int, help="set 1 to use gpu."
# )
# args = parser.parse_known_args()

# set global torch tensor variables. default is using cpu
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor