from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils
import os
from six.moves import cPickle
import numpy as np

from .CaptionModel import CaptionModel

class LSTMCore(nn.Module):
    def __init__(self, opt):
        super(LSTMCore, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm

        # Build a LSTM
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

    def forward(self, xt, state):

        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        in_transform = torch.max(\
            all_input_sums.narrow(1, 3 * self.rnn_size, self.rnn_size),
            all_input_sums.narrow(1, 4 * self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)

        output = self.dropout(next_h)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state

class FCModel_binary(CaptionModel):
    def __init__(self, opt):
        super(FCModel_binary, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size

        self.ss_prob = 0.0 # Schedule sampling probability

        self.img_embed = nn.Linear(self.fc_feat_size, self.input_encoding_size)
        self.core = LSTMCore(opt)
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size)

        self.init_weights()
        with open(os.path.join(opt.binary_tree_coding_dir, 'binary_tree_coding.pkl')) as f:
            binary_tree_coding = cPickle.load(f)
        self.depth = binary_tree_coding['depth']
        self.vocab2code = binary_tree_coding['vocab2code']
        self.phi_list = binary_tree_coding['phi_list']
        self.stop_list = binary_tree_coding['stop_list']
        self.code2vocab = binary_tree_coding['code2vocab']

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'lstm':
            return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                    weight.new_zeros(self.num_layers, bsz, self.rnn_size))
        else:
            return weight.new_zeros(self.num_layers, bsz, self.rnn_size)

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        # sequence is padded with zero in the beginning
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        outputs = []

        for i in range(seq.size(1)):
            if i == 0:
                xt = self.img_embed(fc_feats)
            else:
                it = seq[:, i-1].clone()
                # break if all the sequences end
                if i >= 2 and seq[:, i-1].sum() == 0:
                    break
                xt = self.embed(it)

            output, state = self.core(xt, state)
            output = self.logit(output)
            outputs.append(output)

        return torch.cat([_.unsqueeze(1) for _ in outputs[1:]], 1).contiguous()

    def get_logprobs_state(self, it, state):
        # 'it' is contains a word index
        xt = self.embed(it)

        output, state = self.core(xt, state)
        logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, state

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            for t in range(2):
                if t == 0:
                    xt = self.img_embed(fc_feats[k:k+1]).expand(beam_size, self.input_encoding_size)
                elif t == 1: # input <bos>
                    it = fc_feats.data.new(beam_size).long().zero_()
                    xt = self.embed(it)

                output, state = self.core(xt, state)
                logprobs = F.log_softmax(self.logit(output), dim=1)

            self.done_beams[k] = self.beam_search(state, logprobs, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, fc_feats, att_feats, att_masks=None, opt={}, total_probs=False):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        seq = fc_feats.new_zeros(batch_size, self.seq_length, dtype=torch.long).cuda()
        output_logit = fc_feats.new_zeros(batch_size, self.seq_length, self.depth).cuda()
        unfinished = fc_feats.new_ones(batch_size, dtype=torch.uint8).cuda()
        binary_code = fc_feats.new_ones(batch_size, self.seq_length, self.depth, dtype=torch.uint8).cuda()
        for t in range(self.seq_length + 2):
            if t == 0:
                xt = self.img_embed(fc_feats)
            else:
                if t == 1: # input <bos>
                    it = fc_feats.data.new(batch_size).long().zero_()
                xt = self.embed(it)

            output, state = self.core(xt, state)
            phi = self.logit(output) # phi: batch, vocab-1
            phi_exp = torch.exp(phi)
            # sample the next_word
            if t == self.seq_length + 1: # skip if we achieve maximum length
                break
            if t >= 1:
                mask_depth = unfinished.clone()
                code_sum = np.zeros(batch_size)
                for i in range(self.depth):
                    if i == 0: #two level mask ugly!
                        phi_index = torch.zeros(batch_size, 1).long().cuda()
                    if i > 0:
                        if mask_depth.sum() == 0:
                            break
                        phi_index = torch.from_numpy(map_phi(self.phi_list[i],
                                            np.expand_dims(code_sum * unfinished.cpu().numpy(), 1))).long().cuda()
                    phi_exp_depth = phi_exp.gather(1, phi_index)  # batch, 1
                    phi_depth = phi.gather(1, phi_index)
                    if sample_max:
                        pi = 0.5
                    else:
                        pi = torch.from_numpy(np.random.uniform(size=[batch_size, 1])).float().cuda()
                    it_depth = (pi > 1.0 / (1.0 + phi_exp_depth))  # int8
                    binary_code[:, t - 1, i] = it_depth.squeeze(1) * mask_depth
                    output_logit[:, t - 1, i] = (phi_depth * it_depth.type_as(phi_depth) - torch.log(1.0 + phi_exp_depth)).squeeze(1)
                    if len(self.stop_list[i]) != 0:
                        mask_depth *= torch.from_numpy(unfinished_fun(code_sum, self.stop_list[i])).cuda().type_as(mask_depth)
                    code_sum += (it_depth.squeeze(1) * mask_depth).cpu().numpy() * np.power(2, i)  # batch
                it = torch.from_numpy(code2vocab_fun(code_sum, self.code2vocab)).cuda().long()
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                it = it * unfinished.type_as(it)
                seq[:, t-1] = it
                if unfinished.sum() == 0:
                    break
        return seq, output_logit


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

def unfinished_fun(code_sum, stop_list_i):
    # code_sum: batch, stop_list_i: list of code_sum that should stop
    batch_size = np.shape(code_sum)[0]
    unfinished = np.zeros(batch_size)
    for i in range(batch_size):
        unfinished[i] = np.sum(code_sum[i] == np.array(stop_list_i)) == 0
    return unfinished

def code2vocab_fun(code_sum, code2vocab):
    batch_size = np.shape(code_sum)[0]
    vocab_return = np.zeros(batch_size)
    for i in range(batch_size):
        vocab_return[i] = code2vocab[code_sum[i]]
    return vocab_return