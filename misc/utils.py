from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
def if_use_att(caption_model):
    # Decide if load attention feature according to caption model
    if caption_model in ['show_tell', 'all_img', 'fc']:
        return False
    return True

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        out.append(txt)
    return out

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]

        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class LanguageModelCriterion_binary(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion_binary, self).__init__()

    def forward(self, input, target, mask, depth, vocab2code, phi_list):
        # input: batch, length, vocab-1, cuda
        # target: batch, length, cuda
        # mask: batch, length, cuda,
        # vocab2code: numpy, vocab - 1, depth
        # phi_list: dict
        batch_size, length, _ = input.size()
        target = target[:, :length]
        mask = mask[:, :length].float()
        # not right
        code = vocab2code[target.cpu().numpy(), :].copy()  # batch, length, depth, numpy
        code_sum = np.cumsum(code * np.power(2, np.arange(depth)), 2)  # batch, length, depth
        loss = torch.zeros([]).float().cuda()
        mask_sum = 0
        #TODO: fix log_softmax
        for i in range(depth):
            if i == 0:
                phi_index = torch.zeros(batch_size, length, 1).long().cuda()
                output_logit = input.gather(2, phi_index)
                mask_step = mask.clone()
                mask_sum += mask_step.sum()
                loss -= ((output_logit.squeeze(2) * torch.from_numpy(code[:, :, i]).cuda().float() -
                         LogOnePlusExp(output_logit.squeeze(2))) * mask_step).sum()
            else:
                if np.sum(code[:, :, i] == -1) == batch_size * length:
                    break
                #TODO: efficient map phi
                phi_index = map_phi(phi_list[i], code_sum[:, :, i - 1] * (code[:, :, i] >= 0))  # batch, length
                output_logit = input.gather(2, torch.from_numpy(phi_index).long().cuda().unsqueeze(2))  # input_logit: batch, length, V-1, gather: batch, length
                mask_step = torch.from_numpy((code[:, :, i] >= 0).astype(int)).cuda().float() * mask_step
                mask_sum += mask_step.sum()
                loss -= ((output_logit.squeeze(2) * torch.from_numpy(code[:, :, i]).cuda().float() -
                          LogOnePlusExp(output_logit.squeeze(2))) * mask_step).sum()
        loss = loss / mask.sum()

        return loss


class LanguageModelCriterion_binary_2_layer(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion_binary_2_layer, self).__init__()

    def forward(self, input, target, mask, depth, vocab2code, phi_list):
        # input: batch, length, vocab-1, cuda
        # target: batch, length, cuda
        # mask: batch, length, cuda,
        # vocab2code: numpy, vocab - 1, depth
        # phi_list: dict
        batch_size, length, vocab_1 = input.size()
        vocab = vocab_1 + 1  # 9488
        target = target[:, :length]
        mask = mask[:, :length].float()
        # not right
        code = vocab2code[target.cpu().numpy(), :].copy()  # batch, length, 2, numpy
        code_sum = np.sum(vocab2code * [vocab, 1], 1)  # batch, length, depth
        loss = torch.zeros([]).float().cuda()
        mask_sum = 0
        cluster1_size = int(np.sum(vocab2code[:, 0]))
        cluster0_size = int(vocab - cluster1_size)
        # first layer
        phi_index = torch.zeros(batch_size, length, 1).long().cuda()
        output_logit = input.gather(2, phi_index)
        mask_step = mask
        mask_sum += mask_step.sum()
        loss -= ((output_logit.squeeze(2) * torch.from_numpy(code[:, :, 0]).cuda().float() -
                         LogOnePlusExp(output_logit.squeeze(2))) * mask_step).sum()
        cluster1_mask = (torch.from_numpy(code[:, :, 0]) == 1).float().cuda()
        cluster0_mask = (torch.from_numpy(code[:, :, 0]) == 0).float().cuda()
        logits_cluster1 = F.log_softmax(torch.cat([input[:,:,1:(cluster1_size)], #batch, length, cluster1_size
                                                torch.ones(batch_size, length, 1).float().cuda()], 2), 2)

        logits_cluster0 = F.log_softmax(torch.cat([input[:,:,cluster1_size:],  #batch, length, cluster0_size
                                                torch.ones(batch_size, length, 1).float().cuda()], 2), 2)
        code_2_cuda = torch.from_numpy(code[:, :, 1:2]).long().cuda()
        loss -= (logits_cluster0.gather(
            2, torch.min(code_2_cuda, (cluster0_size-1)*torch.ones_like(code_2_cuda).long().cuda())).squeeze(2) * cluster0_mask * mask).sum()
        loss -= (logits_cluster1.gather(
            2, torch.min(code_2_cuda, (cluster1_size-1)*torch.ones_like(code_2_cuda).long().cuda())).squeeze(2) * cluster1_mask * mask).sum()

        return loss / mask.sum()


def LogOnePlusExp(x):
    result = torch.zeros_like(x).float().cuda()
    result[x > 0] = (x[x > 0] + torch.log1p(torch.exp(-x[x > 0])))
    result[x <= 0] = torch.log1p(torch.exp(x[x <= 0]))
    return result

def map_phi(phi, code_sum):
    # phi: a dictionary,
    # code_sum, a matrix containing code_sum: batch, length
    phi_index = np.zeros_like(code_sum)
    batch, length = np.shape(code_sum)
    if len(phi) < batch * length:
        for i in phi:
            phi_index[code_sum == i] = phi[i]
    else:
        for i in range(batch):
            for j in range(length):
                if phi.get(code_sum[i, j]) is not None:
                    phi_index[i, j] = phi[code_sum[i, j]]
                else:
                    phi_index[i, j] = 0
    return phi_index


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def build_optimizer(params, opt):
    if opt.optim == 'rmsprop':
        return optim.RMSprop(params, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, weight_decay=opt.weight_decay)
    elif opt.optim == 'adagrad':
        return optim.Adagrad(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdm':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdmom':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay, nesterov=True)
    elif opt.optim == 'adam':
        return optim.Adam(params, opt.learning_rate, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon, weight_decay=opt.weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))
