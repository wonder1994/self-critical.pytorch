from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from time import time
import misc.utils as utils
from collections import OrderedDict
import torch
import torch.nn.functional as F
import misc.utils as utils

import sys
sys.path.append("/home1/06008/xf993/self-critical.pytorch/cider")
from pyciderevalcap.ciderD.ciderD import CiderD
sys.path.append("/home1/06008/xf993/self-critical.pytorch/coco-caption")
from pycocoevalcap.bleu.bleu import Bleu

CiderD_scorer = None
Bleu_scorer = None
#CiderD_scorer = CiderD(df='corpus')

def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

def get_self_critical_reward(model, fc_feats, att_feats, att_masks, data, gen_result, opt):
    batch_size = gen_result.size(0)# batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])
    
    # get greedy decoding baseline
    model.eval()
    with torch.no_grad():
        greedy_res, _ = model(fc_feats, att_feats, att_masks=att_masks, mode='sample')
    model.train()

    res = OrderedDict()
    
    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(2 * batch_size)]
    res__ = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_)
        print('Cider scores:', _)
    else:
        cider_scores = 0
    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
        bleu_scores = np.array(bleu_scores[3])
        print('Bleu scores:', _[3])
    else:
        bleu_scores = 0
    scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores

    scores = scores[:batch_size] - scores[batch_size:]

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards


def get_reward(data, gen_result, opt):
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])

    res = OrderedDict()

    gen_result = gen_result.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    res_ = [{'image_id': i, 'caption': res[i]} for i in range(batch_size)]
    res__ = {i: res[i] for i in range(batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(batch_size)}
    print(gen_result[0])
    print(gts[0])
    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_)
        print('Cider scores:', _)
    else:
        cider_scores = 0
    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
        bleu_scores = np.array(bleu_scores[3])
        print('Bleu scores:', _[3])
    else:
        bleu_scores = 0
    scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)
    if opt.rf_demean == 1:
        rewards = np.repeat(scores[:, np.newaxis] - np.mean(scores[:, np.newaxis]), gen_result.shape[1], 1)

    return rewards


def get_arm_loss(model, fc_feats, att_feats, att_masks, data, opt, loader):
    batch_size = fc_feats.size(0)
    vocab_size = opt.vocab_size + 1
    state = model.init_hidden(batch_size)
    seq = fc_feats.new_zeros(batch_size, model.seq_length, dtype=torch.long)
    loss = fc_feats.new_zeros([])
    unfinished = fc_feats.new_ones(batch_size, dtype=torch.uint8)
    ticin = time()
    for t in range(model.seq_length + 1):
        if t == 0:
            xt = model.img_embed(fc_feats)
        else:
            if t == 1:
                it = fc_feats.data.new(batch_size).long().zero_()
            xt = model.embed(it)

        output, state = model.core(xt, state)
        #print(opt.seq_per_img)
        if t >= 1:
            logprobs = F.log_softmax(model.logit(output), dim=1)
            logprobs_numpy = logprobs.detach().cpu().numpy()
            pi = np.random.dirichlet(np.ones(vocab_size), batch_size)
            f_delta = arsm_f_delta_fun_batch(logprobs_numpy, pi, data, seq, t, model, state, unfinished, loader)
            mask = unfinished.float()
            f_delta = torch.from_numpy(f_delta).float().cuda()
            f_delta = (f_delta.transpose(0, 1) * mask).transpose(0, 1) / torch.sum(mask)
            loss -= torch.sum(f_delta * logprobs)

            it = torch.from_numpy(np.argmin(np.exp(-logprobs_numpy) * pi, axis=1)).cuda()
            it = it.view(-1).long()

            if t == 1:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)

            it = it * unfinished.type_as(it)
            seq[:, t-1] = it
            if unfinished.sum() == 0:
                break
    #print('*******************************************time for one iteration: ' + str(time()-ticin))
    return loss


def arsm_f_delta_fun_batch(logits, pi, data, pre_seq, step, model, state, unfinished, loader, type='ars', print_pseudo=True):
    #TODO: write in torch
    batch_size, vocab_size = np.shape(logits)
    index_batch = np.arange(batch_size)
    index_vocab = np.arange(vocab_size)
    ## pseudo actions (in numpy):
    tic = time()
    exp_neg_logit = np.exp(-logits)
    min_value = np.repeat(np.expand_dims(np.min(pi * exp_neg_logit, axis=1), 1), vocab_size, 1)
    A_cat = np.argmin(pi * exp_neg_logit, axis=1)
    R_cat = np.random.randint(vocab_size, size=batch_size)
    pseudo_actions = np.repeat(np.expand_dims(A_cat, 1), vocab_size, axis=1)
    pseudo_actions += np.less(exp_neg_logit * np.repeat(
        np.expand_dims(pi[index_batch, R_cat], 1), vocab_size, 1), min_value) * (index_vocab - np.expand_dims(A_cat, 1))
    pseudo_actions += np.less(pi * np.repeat(
        np.expand_dims(exp_neg_logit[index_batch, R_cat], 1), vocab_size, 1), min_value) * \
                      np.repeat(np.expand_dims(R_cat - A_cat, 1), vocab_size, 1)
    top_2_indices = (pi * exp_neg_logit).argsort()[:, 1]
    top_2_values = np.repeat(np.expand_dims((pi * exp_neg_logit)[index_batch, top_2_indices], 1), vocab_size, 1)
    candidate_i_value = exp_neg_logit * np.repeat(np.expand_dims(pi[index_batch, A_cat], 1), vocab_size, 1)
    candidate_A_value = pi * np.repeat(np.expand_dims(exp_neg_logit[index_batch, A_cat], 1), vocab_size, 1)
    pseudo_actions_true = np.repeat(np.expand_dims(top_2_indices, 1), vocab_size, 1)
    pseudo_actions_true += np.less(candidate_i_value, top_2_values) * np.less(candidate_i_value, candidate_A_value) * \
                           (index_vocab - np.expand_dims(top_2_indices, axis=1))
    pseudo_actions_true += np.less(candidate_A_value, top_2_values) * np.less(candidate_A_value, candidate_i_value) * \
                           np.repeat(np.expand_dims(A_cat - top_2_indices, axis=1), vocab_size, 1)
    index_matrix = np.zeros_like(logits)
    index_matrix[index_batch, A_cat] = 1
    index_matrix[R_cat == A_cat, :] = 1
    pseudo_actions = pseudo_actions + index_matrix * (pseudo_actions_true - pseudo_actions)
    #print('time for pseudo action: ' + str(time() - tic))
    tic = time()
    ## concate unique pseudo actions
    arm_pseudo_action_set = []
    arm_index = []
    arm_index_2 = np.zeros(0)
    arm_pseudo_counts = []
    for i in range(batch_size):
        set_per_sample, index_per_sample = np.unique(pseudo_actions[i, :], return_inverse=True)
        pseudo_count = len(set_per_sample)
        arm_pseudo_counts.append(pseudo_count)
        arm_pseudo_action_set = np.concatenate([arm_pseudo_action_set, set_per_sample], axis=0)
        arm_index.append(index_per_sample)
        arm_index_2 = np.concatenate([arm_index_2, (np.ones(pseudo_count) * i)], axis=0)
    #print('time for concatenation: ' + str(time() - tic))
    ## complete sentences
    tic= time()
    seqs_arm = pre_seq[arm_index_2, :]
    unfinished_arm = unfinished[arm_index_2]
    seqs_arm[:, step-1] = arm_pseudo_action_set * unfinished_arm
    state_h, state_c = state
    state_h_arm = state_h[:, arm_index_2, :]
    state_c_arm = state_c[:, arm_index_2, :]
    state_arm = (state_h_arm, state_c_arm)
    it = torch.from_numpy(arm_pseudo_action_set).long().cuda()
    for t in range(step + 1, model.seq_length + 1):
        if unfinished_arm.sum() == 0:
            break
        xt = model.embed(it)
        output, state_arm = model.core(xt, state_arm)
        logprobs = F.log_softmax(model.logit(output), dim=1)
        prob_prev = torch.exp(logprobs.data).cpu()
        it = torch.multinomial(prob_prev, 1).cuda()
        it = it.view(-1).long()
        unfinished_arm = (it > 0) * unfinished_arm
        seqs_arm[:, t-1] = it * unfinished_arm.type_as(it)
    #print('time for completion: ' + str(time() - tic))
    ## evaluate reward
    tic = time()
    seq_per_img = batch_size // len(data['gts'])
    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]
    seqs_arm = seqs_arm.data.cpu().numpy()
    sents = utils.decode_sequence(loader.get_vocab(), seqs_arm)
    if print_pseudo and step == np.random.randint(20):
        print('**********************At step ' + str(step))
        print(sents[0:arm_pseudo_counts[0]])
    res_ = []
    gts_arm = {}
    cum_count = np.cumsum(arm_pseudo_counts)
    for i in range(len(arm_pseudo_action_set)):
        res_.append({'image_id': i, 'caption': [array_to_str(seqs_arm[i])]})
        i_index = np.min(np.nonzero(np.less(i, cum_count)))
        gts_arm[i] = gts[i_index // seq_per_img]
    #print('time for prepare reward:' + str(time() - tic))
    tic = time()
    _, arm_metric_value = CiderD_scorer.compute_score(gts_arm, res_)
    print('Cider value:', _)
    arm_index = np.array(arm_index)
    arm_index += np.repeat(np.expand_dims(np.concatenate([[0], np.cumsum(arm_pseudo_counts)[0:(batch_size-1)]]), 1), vocab_size, 1)
    arm_index = np.reshape(arm_index, [-1])
    #print('time for evaluating pseudo action: ' + str(time() - tic))
    #print(arm_metric_value)
    arm_metric_matrix = np.reshape(arm_metric_value[arm_index], [batch_size, vocab_size])
    f_delta = arm_metric_matrix - np.repeat(np.expand_dims(np.mean(arm_metric_matrix, 1), 1), vocab_size, 1)
    f_delta = f_delta * np.repeat(np.expand_dims(1.0 / vocab_size - pi[index_batch, R_cat], 1), vocab_size, 1)
    return f_delta


def arsm_f_delta_fun(logits, pi_batch, data, pre_seq, step, model, state, unfinished, type='ars'):
    batch_size, vocab_size = np.shape(logits)
    f_delta = np.zeros_like(logits)
    seq_per_img = batch_size // len(data['gts'])
    #TODO: parallel, look at pseudo action number, look at the memory
    for i in range(batch_size):
        if unfinished[i]:
            pi = pi_batch[i, :]
            phi_arsm = logits[i, :]
            action_true = np.argmin(np.log(pi) - phi_arsm)
            target = [array_to_str(data['gts'][i // seq_per_img][j]) for j in range(len(data['gts'][i // seq_per_img]))]
            state_h, state_c = state
            state_h_i = state_h[:, i, :].unsqueeze(1)
            state_c_i = state_c[:, i, :].unsqueeze(1)
            state_i = (state_h_i, state_c_i)
            if type == 'arsm':
                pseudo_actions = pseudo_action_swap_matrix(pi, phi_arsm)
                unique_pseudo_actions = np.unique(pseudo_actions[pseudo_actions != action_true])
                F = np.full((vocab_size, vocab_size),
                            reward_func(torch.from_numpy(np.expand_dims(action_true, 0)), pre_seq[i].unsqueeze(0),
                                        target, step, model, state_i))
                for action in unique_pseudo_actions:
                    a = reward_func(torch.from_numpy(np.expand_dims(action, 0)), pre_seq[i].unsqueeze(0), target, step,
                                    model, state_i)
                    F[pseudo_actions == action] = a
                meanF = np.mean(F, axis=0)
                f_delta[i, :] = np.matmul(F - meanF, 1.0 / vocab_size - pi)
            elif type == 'ars':
                ref_cat = np.argmin(-phi_arsm)
                tic = time()
                pseudo_actions = pseudo_action_swap_vector(pi, phi_arsm, ref_cat)
                #print("pseudo action swap time: " + str(time() - tic))
                tic = time()
                unique_pseudo_actions = np.unique(pseudo_actions[pseudo_actions != action_true])
                #print('pseudo action:' + str(np.shape(unique_pseudo_actions)[0]))
                F = np.full(vocab_size, reward_func(torch.from_numpy(np.expand_dims(action_true, 0)), pre_seq[i].unsqueeze(0), target, step, model, state_i))
                for action in unique_pseudo_actions:
                    tic1 = time()
                    a = reward_func(torch.from_numpy(np.expand_dims(action, 0)), pre_seq[i].unsqueeze(0), target, step, model, state_i)
                    #print('reward time'+ str(time()-tic1))
                    F[pseudo_actions == action] = a
                meanF = np.mean(F, axis=0)
                f_delta[i, :] = (F - meanF) * (1.0 / vocab_size - pi[ref_cat])
                #print("reward evaluation time: " + str(time() - tic))
    return f_delta


def reward_func(action, pre_seq, target, step, model, state):
    pre_seq[:, step-1] = action
    unfinished = (action > 0).cuda()
    it = action.cuda()
    tic = time()
    ## complete sentence
    for t in range(step + 1, model.seq_length + 1):
        if unfinished.sum() == 0:
            break
        xt = model.embed(it)
        output, state = model.core(xt, state)
        logprobs = F.log_softmax(model.logit(output), dim=1)
        prob_prev = torch.exp(logprobs.data).cpu()
        it = torch.multinomial(prob_prev, 1).cuda()
        it = it.view(-1).long()
        unfinished = (it > 0) * unfinished
        pre_seq[:, t-1] = it * unfinished.type_as(it)
    #print('time for completion: ' + str(time() - tic))
    ## evaluate reward
    pre_seq = pre_seq.data.cpu().numpy()
    res_ = [{'image_id': 0, 'caption': [array_to_str(pre_seq[0])]}]
    gts = {0: target}
    tic = time()
    _, cider_scores = CiderD_scorer.compute_score(gts, res_)
    print(cider_scores)
    #print('time for eval: ' + str(time() - tic))
    return _



def pseudo_action_swap_matrix(pi, phi):
    C = len(pi)
    RaceAllSwap = np.log(pi[:, np.newaxis]) - phi[np.newaxis, :]
    Race = np.diag(RaceAllSwap)
    action_true = np.argmin(Race)
    Race_min = Race[action_true]

    if C < 7:
        # Slow version for large C
        pseudo_actions = np.full((C, C), action_true)
        for m in range(C):
            for jj in range(m):
                RaceSwap = Race.copy()
                RaceSwap[m], RaceSwap[jj] = RaceAllSwap[jj, m], RaceAllSwap[m, jj]
                s_action = np.argmin(RaceSwap)
                pseudo_actions[m, jj], pseudo_actions[jj, m] = s_action, s_action
    else:
        # Fast version for large C
        pseudo_actions = np.full((C, C), action_true)

        SwapSuccess = RaceAllSwap <= Race_min
        SwapSuccess[action_true, :] = True
        np.fill_diagonal(SwapSuccess, 0)
        m_idx, j_idx = np.where(SwapSuccess)

        for i in range(len(m_idx)):
            m, jj = m_idx[i], j_idx[i]
            RaceSwap = Race.copy()
            RaceSwap[m], RaceSwap[jj] = RaceAllSwap[jj, m], RaceAllSwap[m, jj]
            if m == action_true or jj == action_true:
                s_action = np.argmin(RaceSwap)
                pseudo_actions[m, jj], pseudo_actions[jj, m] = s_action, s_action
            else:
                if RaceSwap[m] < RaceSwap[jj]:
                    pseudo_actions[m, jj], pseudo_actions[jj, m] = m, m
                else:
                    pseudo_actions[m, jj], pseudo_actions[jj, m] = jj, jj

    return pseudo_actions


def pseudo_action_swap_vector(pi,phi,Cat_ref):
    C=len(pi)
    Race = np.log(pi)-phi
    action_true=np.argmin(Race)
    min_value = Race[action_true]
    jj = Cat_ref
    pseudo_actions=np.full(C, action_true)
    for m in range(C):
        if m == action_true or Cat_ref == action_true:
            RaceSwap = Race.copy()
            RaceSwap[m] = np.log(pi[jj]) - phi[m]
            RaceSwap[jj] = np.log(pi[m]) - phi[jj]
            pseudo_actions[m] = np.argmin(RaceSwap)

        else:
            if np.min([np.log(pi[jj])-phi[m], np.log(pi[m]) - phi[jj]]) < min_value:
                if np.log(pi[jj]) - phi[m] < np.log(pi[m]) - phi[jj]:
                    pseudo_actions[m] = m
                else:
                    pseudo_actions[m] = jj
    return pseudo_actions


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()