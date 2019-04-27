import torch
import numpy as np
from time import time
def pseudo_action_fun(logits, A_cat, R_cat, pi, temperature=1):
    #TODO: log pi.
    batch_size, vocab_size = logits.size()
    index_batch = torch.arange(batch_size).cuda().long()
    index_vocab = torch.arange(vocab_size).cuda().long()
    min_value = torch.min(torch.log(pi) - logits, 1)[0].unsqueeze(1).repeat(1, vocab_size)
    pseudo_actions = A_cat.unsqueeze(1).repeat(1, vocab_size)
    pseudo_actions += ((-logits + torch.log(pi[index_batch, R_cat]).unsqueeze(1).repeat(1, vocab_size)) < min_value).long() * \
                      (index_vocab - A_cat.unsqueeze(1))
    pseudo_actions += ((torch.log(pi) - logits[index_batch, R_cat].unsqueeze(1).repeat(1, vocab_size)) < min_value).long() * \
                      (R_cat - A_cat).unsqueeze(1).repeat(1, vocab_size)
    index_matrix = torch.zeros_like(logits).long()
    index_matrix[index_batch, A_cat] = 1
    index_matrix[R_cat == A_cat, :] = 1

    topk, indices = torch.topk(-(torch.log(pi) - logits), 2, dim=1)
    top_2_indices = indices[:, 1]
    top_2_values = -topk[:, 1].unsqueeze(1).repeat(1, vocab_size)
    candidate_i_value = -logits + torch.log(pi[index_batch, R_cat]).unsqueeze(1).repeat(1, vocab_size)
    candidate_A_value = torch.log(pi) - logits[index_batch, R_cat].unsqueeze(1).repeat(1, vocab_size)
    pseudo_actions_true = top_2_indices.unsqueeze(1).repeat(1, vocab_size)
    pseudo_actions_true += (candidate_i_value < top_2_values).long() * (candidate_i_value <= candidate_A_value).long() * \
                           (index_vocab - top_2_indices.unsqueeze(1))
    pseudo_actions_true += (candidate_A_value < top_2_values).long() * (candidate_A_value < candidate_i_value).long() * \
                           (R_cat - top_2_indices).unsqueeze(1).repeat(1, vocab_size)

    pseudo_actions = pseudo_actions + index_matrix * (pseudo_actions_true - pseudo_actions)
    return pseudo_actions

batch_size = 50
vocab_size = 100
logits = torch.from_numpy(np.random.normal(size=[batch_size,vocab_size])).float().cuda()
pi =  torch.from_numpy(np.random.dirichlet(np.ones(vocab_size), batch_size)).float().cuda()
random_ref = torch.topk(torch.rand((batch_size, vocab_size)), 2)[1].cuda().long()
A_cat = torch.from_numpy(np.random.randint(vocab_size, size=[50])).long().cuda()
R_cat = random_ref[:, 0]
tic = time()
results = pseudo_action_fun(logits,A_cat,R_cat,pi)
print('time:', time()-tic)




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

tic = time()
results = pseudo_action_swap_matrix(logits.cpu().numpy()[0,:], pi.cpu().numpy()[0,:])
print('time:', time()-tic)