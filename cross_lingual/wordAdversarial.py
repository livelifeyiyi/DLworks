"""
# ref: MUSE from facebook
build_word_adversarial_model
class WordAdversarialTrainer
EmbeddingMapping
EmbeddingProj
WordDiscriminator
"""
import os
import numpy as np
import scipy
import scipy.linalg

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F

from muse.muse_utils import clip_parameters, load_embeddings, export_embeddings
from muse.muse_utils import normalize_embeddings
from muse.muse_dico_builder import build_dictionary


def build_word_adversarial_model(params):
    # prepare data for word level adversarial
    target_dico, _target_emb = load_embeddings(params, target=True)
    params['target_dico'] = target_dico
    target_emb = nn.Embedding(len(target_dico), params['word_dim'], sparse=True)
    target_emb.weight.data.copy_(_target_emb)

    # target embeddings
    related_dico, _related_emb = load_embeddings(params, target=False)
    params['related_dico'] = related_dico
    related_emb = nn.Embedding(len(related_dico), params['word_dim'], sparse=True)
    related_emb.weight.data.copy_(_related_emb)

    params['target_mean'] = normalize_embeddings(target_emb.weight.data, params['normalize_embeddings'])
    params['related_mean'] = normalize_embeddings(related_emb.weight.data, params['normalize_embeddings'])

    # embedding projection function
    embedding_mapping = EmbeddingMapping(params['word_dim'], params['word_dim'])
    # define the word discriminator
    word_discriminator = WordDiscriminator(input_dim=params['word_dim'],
                                           hidden_dim=params['dis_hid_dim'],
                                           dis_layers=params['dis_layers'],
                                           dis_input_dropout=params['dis_input_dropout'],
                                           dis_dropout=params['dis_dropout'])

    if params['gpu']:
        target_emb = target_emb.cuda()
        related_emb = related_emb.cuda()
        embedding_mapping = embedding_mapping.cuda()
        word_discriminator = word_discriminator.cuda()

    return target_emb, related_emb, embedding_mapping, word_discriminator


class WordAdversarialTrainer(object):
    def __init__(self, target_emb, related_emb, mapping, discriminator, params):
        """
        Initialize trainer script.
        """
        self.target_emb = target_emb
        self.related_emb = related_emb
        self.target_dico = params['target_dico']
        self.related_dico = params['related_dico']
        self.mapping = mapping
        self.discriminator = discriminator
        self.params = params

        # optimizers
        map_optim_fn = optim.SGD
        if params['map_optimizer'] == 'adadelta':
            map_optim_fn = optim.Adadelta
        elif params['map_optimizer'] == 'adagrad':
            map_optim_fn = optim.Adagrad
        elif params['map_optimizer'] == 'adam':
            map_optim_fn = optim.Adam
        elif params['map_optimizer'] == 'adamax':
            map_optim_fn = optim.Adamax
        elif params['map_optimizer'] == 'asgd':
            map_optim_fn = optim.ASGD
        elif params['map_optimizer'] == 'rmsprop':
            map_optim_fn = optim.RMSprop
        elif params['map_optimizer'] == 'rprop':
            map_optim_fn = optim.Rprop
        elif params['map_optimizer'] == 'sgd':
            map_optim_fn = optim.SGD

        self.map_optimizer = map_optim_fn(mapping.parameters(), lr=params['map_learning_rate'], momentum=0.9)

        dis_optim_fn = optim.SGD
        if params['dis_optimizer'] == 'adadelta':
            dis_optim_fn = optim.Adadelta
        elif params['dis_optimizer'] == 'adagrad':
            dis_optim_fn = optim.Adagrad
        elif params['dis_optimizer'] == 'adam':
            dis_optim_fn = optim.Adam
        elif params['dis_optimizer'] == 'adamax':
            dis_optim_fn = optim.Adamax
        elif params['dis_optimizer'] == 'asgd':
            dis_optim_fn = optim.ASGD
        elif params['dis_optimizer'] == 'rmsprop':
            dis_optim_fn = optim.RMSprop
        elif params['dis_optimizer'] == 'rprop':
            dis_optim_fn = optim.Rprop
        elif params['dis_optimizer'] == 'sgd':
            dis_optim_fn = optim.SGD

        self.dis_optimizer = dis_optim_fn(discriminator.parameters(), lr=params['dis_learning_rate'], momentum=0.9)

        # best validation score
        self.best_valid_metric = -1e12
        self.decrease_lr = False

    def dis_step(self, target_ids, related_ids):
        self.discriminator.train()

        # get word embeddings
        target_emb_tmp = self.target_emb(Variable(target_ids, volatile=True))
        related_emb_tmp = self.related_emb(Variable(related_ids, volatile=True))
        related_emb_tmp = self.mapping(Variable(related_emb_tmp.data, volatile=True))
        target_emb_tmp = Variable(target_emb_tmp.data, volatile=True)

        batch_size = target_ids.size(0)

        # input / target
        x = torch.cat([related_emb_tmp, target_emb_tmp], 0)
        y = torch.FloatTensor(batch_size * 2).zero_()
        y[:batch_size] = 1 - self.params['dis_smooth']
        y[batch_size:] = self.params['dis_smooth']
        y = Variable(y.cuda() if self.params['gpu'] else y)

        # loss
        preds = self.discriminator(Variable(x.data))
        loss = F.binary_cross_entropy(preds, y)

        # check NaN
        if (loss != loss).data.any():
            print("NaN detected (discriminator)")
            exit()

        # optim
        self.dis_optimizer.zero_grad()
        loss.backward()
        self.dis_optimizer.step()
        clip_parameters(self.discriminator, 0) # self.params['dis_clip_weights']
        return loss

    def mapping_step(self, target_ids, related_ids):
        """
        Fooling discriminator training step.
        """
        if self.params['dis_lambda'] == 0:
            return 0

        self.discriminator.eval()

        # get word embeddings
        target_emb_tmp = self.target_emb(Variable(target_ids, volatile=True))
        related_emb_tmp = self.related_emb(Variable(related_ids, volatile=True))
        related_emb_tmp = self.mapping(Variable(related_emb_tmp.data, volatile=True))
        target_emb_tmp = Variable(target_emb_tmp.data, volatile=True)

        batch_size = target_ids.size(0)

        # input / target
        x = torch.cat([related_emb_tmp, target_emb_tmp], 0)
        y = torch.FloatTensor(batch_size * 2).zero_()
        y[:batch_size] = 1 - self.params['dis_smooth']
        y[batch_size:] = self.params['dis_smooth']
        y = Variable(y.cuda() if self.params['gpu'] else y)

        # loss
        preds = self.discriminator(x)
        loss = F.binary_cross_entropy(preds, 1 - y)
        loss = self.params['dis_lambda'] * loss

        # check NaN
        if (loss != loss).data.any():
            print("NaN detected (fool discriminator)")
            exit()

        # optim
        self.map_optimizer.zero_grad()
        loss.backward()
        self.map_optimizer.step()
        self.orthogonalize()

        return 2 * self.params['batch_size'], loss

    def orthogonalize(self):
        """
        Orthogonalize the mapping.
        """
        if self.params['map_beta'] > 0:
            W = self.mapping.mapper.weight.data
            beta = self.params['map_beta']
            W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))

    def update_lr(self, to_log, metric):
        """
        Update learning rate when using SGD.
        """
        if 'sgd' not in self.params['map_optimizer']:
            return
        old_lr = self.map_optimizer.param_groups[0]['lr']
        new_lr = max(self.params['min_lr'], old_lr * self.params['lr_decay'])
        if new_lr < old_lr:
            print("Decreasing learning rate: %.8f -> %.8f" % (old_lr, new_lr))
            self.map_optimizer.param_groups[0]['lr'] = new_lr

        if self.params.lr_shrink < 1 and to_log[metric] >= -1e7:
            if to_log[metric] < self.best_valid_metric:
                print("Validation metric is smaller than the best: %.5f vs %.5f"
                      % (to_log[metric], self.best_valid_metric))
                # decrease the learning rate, only if this is the
                # second time the validation metric decreases
                if self.decrease_lr:
                    old_lr = self.map_optimizer.param_groups[0]['lr']
                    self.map_optimizer.param_groups[0]['lr'] *= self.params['lr_shrink']
                    print("Shrinking the learning rate: %.5f -> %.5f"
                          % (old_lr, self.map_optimizer.param_groups[0]['lr']))
                self.decrease_lr = True

    def save_best(self, to_log, metric):
        """
        Save the best model for the given validation metric.
        """
        # best mapping for the given validation criterion
        if to_log[metric] > self.best_valid_metric:
            # new best mapping
            self.best_valid_metric = to_log[metric]
            print('* Best value for "%s": %.5f' % (metric, to_log[metric]))
            # save the mapping
            W = self.mapping.mapper.weight.data.cpu().numpy()
            path = os.path.join(self.params.exp_path, 'best_mapping.pth')
            print('* Saving the mapping to %s ...' % path)
            torch.save(W, path)

    def reload_best(self):
        """
        Reload the best mapping.
        """
        path = os.path.join(self.params['exp_path'], 'best_mapping.pth')
        print('* Reloading the best model from %s ...' % path)
        # reload the model
        assert os.path.isfile(path)
        to_reload = torch.from_numpy(torch.load(path))
        W = self.mapping.mapper.weight.data
        assert to_reload.size() == W.size()
        W.copy_(to_reload.type_as(W))

    def export(self):
        """
        Export embeddings.
        """
        params = self.params

        # load all embeddings
        print("Reloading all embeddings for mapping ...")
        params['target_dico'], target_emb = load_embeddings(params, target=True, full_vocab=True)
        params['related_dico'], related_emb = load_embeddings(params, target=False, full_vocab=True)

        # apply same normalization as during training
        normalize_embeddings(target_emb, params['normalize_embeddings'], mean=params['target_mean'])
        normalize_embeddings(related_emb, params['normalize_embeddings'], mean=params['related_mean'])

        # map source embeddings to the target space
        bs = 4096
        print("Map source embeddings to the target space ...")
        for i, k in enumerate(range(0, len(related_emb), bs)):
            x = Variable(related_emb[k:k + bs], volatile=True)
            related_emb[k:k + bs] = self.mapping(x.cuda() if params.cuda else x).data.cpu()

        # write embeddings to the disk
        export_embeddings(target_emb, related_emb, params)

    def build_dictionary(self):
        """
        Build a dictionary from aligned embeddings.
        """
        related_emb = self.mapping(self.related_emb.weight).data
        target_emb = self.target_emb.weight.data
        related_emb = related_emb / related_emb.norm(2, 1, keepdim=True).expand_as(related_emb)
        target_emb = target_emb / target_emb.norm(2, 1, keepdim=True).expand_as(target_emb)
        self.dico = build_dictionary(related_emb, target_emb, self.params)

    def procrustes(self):
        """
        Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        """
        A = self.related_emb.weight.data[self.dico[:, 0]]
        B = self.target_emb.weight.data[self.dico[:, 1]]
        W = self.mapping.mapper.weight.data
        M = B.transpose(0, 1).mm(A).cpu().numpy()
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))


class EmbeddingMapping(nn.Module):
    def __init__(self, mono_dim, common_dim):
        super(EmbeddingMapping, self).__init__()
        self.mono_dim = mono_dim
        self.common_dim = common_dim
        self.mapper = nn.Linear(mono_dim, common_dim, bias=False)

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
            else:
                nn.init.xavier_uniform_(p.data)

    def forward(self, input):
        encoded = self.mapper(input)
        return encoded

    def orthogonalize(self, map_beta):
        """
        Orthogonalize the mapping.
        """
        if map_beta > 0:
            W = self.mapper.weight.data
            beta = map_beta
            W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))


class EmbeddingProj(nn.Module):
    def __init__(self, mono_dim, common_dim):
        super(EmbeddingProj, self).__init__()
        self.mono_dim = mono_dim
        self.common_dim = common_dim
        self.encoder = nn.Parameter(torch.Tensor(mono_dim, common_dim))
        self.encoder_bias = nn.Parameter(torch.Tensor(common_dim))
        self.encoder_bn = nn.BatchNorm1d(common_dim, momentum=0.01)

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
            else:
                nn.init.xavier_uniform_(p.data)

    def forward(self, input):
        encoded = torch.sigmoid(
            torch.mm(input, self.encoder) + self.encoder_bias)
        encoded = self.encoder_bn(encoded)
        return encoded


class WordDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, dis_layers, dis_input_dropout, dis_dropout):
        """Init discriminator."""
        super(WordDiscriminator, self).__init__()

        self.emb_dim = input_dim
        self.dis_layers = dis_layers
        self.dis_hid_dim = hidden_dim
        self.dis_dropout = dis_dropout
        self.dis_input_dropout = dis_input_dropout

        layers = [nn.Dropout(self.dis_input_dropout)]

        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
            else:
                nn.init.xavier_uniform_(p.data)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.layers(x).view(-1)

