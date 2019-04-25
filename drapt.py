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
    logits_cluster1 = F.log_softmax(torch.cat([input[:, :, 1:(cluster1_size)],  # batch, length, cluster1_size
                                               torch.ones(batch_size, length, 1).float().cuda()], 2), 2)

    logits_cluster0 = F.log_softmax(torch.cat([input[:, :, cluster1_size:],  # batch, length, cluster0_size
                                               torch.ones(batch_size, length, 1).float().cuda()], 2), 2)
    code_2_cuda = torch.from_numpy(code[:, :, 1:2]).long().cuda()
    loss -= (logits_cluster0.gather(
        2, torch.min(code_2_cuda, (cluster0_size - 1) * torch.ones_like(code_2_cuda).long().cuda())).squeeze(
        2) * cluster0_mask * mask).sum()
    loss -= (logits_cluster1.gather(
        2, torch.min(code_2_cuda, (cluster1_size - 1) * torch.ones_like(code_2_cuda).long().cuda())).squeeze(
        2) * cluster1_mask * mask).sum()

    return loss / mask.sum()