from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
import eval_utils
import misc.utils as utils
from misc.syn_util import init_scorer, get_self_critical_reward, get_reward, get_arm_loss, get_mct_loss, get_ar_loss
from models.CriticModel import CriticModel
from models.AttCriticModel import AttCriticModel, critic_loss_fun, target_critic_loss_fun, target_critic_loss_fun_mask
try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def reward_fun(labels, fc_feats, true_model):
    labels_pad = torch.cat([torch.zeros(labels.size(0), 1).cuda().long(), labels], 1)
    masks = (labels_pad > 0).float()
    logit_reward = true_model(fc_feats, None, labels_pad, None)
    reward = (logit_reward.gather(2, labels_pad[:, 1:].unsqueeze(2)).squeeze(2) * masks[:, 1:]).sum(1)
    return reward

def eval_utils_syn(dp_model, true_model, data_features, batch_size, crit):
    data_num = data_features.shape[0]
    num_batch = int(data_num / batch_size)
    for iteration in range(num_batch):
        start_index = (iteration*batch_size)
        end_index = ((iteration+1)*batch_size)
        fc_feats = torch.from_numpy(data_features[start_index:end_index, :]).cuda().float()
        att_feats = None
        att_masks = None
        true_labels, _ = true_model(fc_feats, att_feats, att_masks, opt={'sample_max':0}, mode='sample')
        #print(true_labels)
        true_labels = torch.cat([torch.zeros(true_labels.size(0), 1).cuda().long(), true_labels], 1)
        masks = (true_labels > 0).float()
        val_loss = crit(dp_model(fc_feats, att_feats, true_labels, att_masks), true_labels[:,1:], masks[:,1:])

        labels, _ = dp_model(fc_feats, att_feats, att_masks, opt={'sample_max':0}, mode='sample')
        labels = torch.cat([torch.zeros(labels.size(0), 1).cuda().long(), labels], 1)
        masks = (labels > 0).float()
        lang_stats = crit(true_model(fc_feats, att_feats, labels, att_masks), labels[:,1:], masks[:,1:])
    return val_loss, lang_stats


def train(opt):
    # opt.use_att = utils.if_use_att(opt.caption_model)
    opt.use_att = True
    if opt.use_box: opt.att_feat_size = opt.att_feat_size + 5

    opt.vocab_size = 50
    opt.seq_length = 10
    opt.fc_feat_size = 100
    opt.train_true = True
    opt.train_true_step = 100
    np.random.seed(0)
    data_num = 5000
    data_features = np.random.normal(size=[data_num, opt.fc_feat_size])
    test_data_num = 1000
    test_data_features = np.random.normal(size=[test_data_num, opt.fc_feat_size])
    print(opt.checkpoint_path)
    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)

    infos = {}
    histories = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')) as f:
                histories = cPickle.load(f)

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    critic_loss_history = histories.get('critic_loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})
    variance_history = histories.get('variance_history', {})
    time_history = histories.get('time_history', {})


    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    model = models.setup(opt).cuda()
    dp_model = model
    #TODO: save true model
    true_model = models.setup(opt).cuda()
    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist
        assert os.path.isdir(opt.start_from)," %s must be a a path" % opt.start_from
        assert os.path.isfile(os.path.join(opt.start_from,"infos_"+opt.id+".pkl")),"infos.pkl file does not exist in path %s"%opt.start_from
        true_model.load_state_dict(torch.load(os.path.join(opt.start_from, 'truemodel.pth')))
    true_model.eval()
    ######################### Actor-critic Training #####################################################################

    update_lr_flag = True
    # Assure in training mode
    dp_model.train()

    crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()

    optimizer = utils.build_optimizer(model.parameters(), opt)
    tm_optimizer = utils.build_optimizer(true_model.parameters(), opt)
    # Load the optimizer
    if vars(opt).get('start_from', None) is not None and os.path.isfile(os.path.join(opt.start_from,"optimizer.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    first_order = 0
    second_order = 0
    while True:
        if update_lr_flag:
                # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate  ** frac
                opt.current_lr = opt.learning_rate * decay_factor
            else:
                opt.current_lr = opt.learning_rate
            utils.set_lr(optimizer, opt.current_lr)
            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob

            # If start self critical training
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                sc_flag = True
                init_scorer(opt.cached_tokens)
            else:
                sc_flag = False

            update_lr_flag = False

        dp_model.train()

        torch.cuda.synchronize()
        start = time.time()
        gen_result = None
        start_index = (iteration*opt.batch_size) % data_num
        end_index = start_index + opt.batch_size
        fc_feats = torch.from_numpy(data_features[start_index:end_index, :]).cuda().float()
        att_feats = None
        att_masks = None
        labels, total_logits = true_model(fc_feats, att_feats, att_masks, opt={'sample_max':1}, total_probs=True, mode='sample')
        labels = torch.cat([torch.zeros(labels.size(0), 1).cuda().long(), labels], 1)
        masks = (labels > 0).float()

        # train true model:
        if iteration < opt.train_true_step and opt.train_true:
            tm_optimizer.zero_grad()
            loss = -((total_logits * F.softmax(total_logits, 2)).sum(2)).mean()
            loss.backward()
            tm_optimizer.step()

        optimizer.zero_grad()
        if not sc_flag:
            loss = crit(dp_model(fc_feats, att_feats, labels, att_masks), labels[:,1:], masks[:,1:])
        else:
            if opt.rl_type == 'sc':
                gen_result, sample_logprobs = dp_model(fc_feats, att_feats, att_masks, opt={'sample_max':0}, mode='sample')
                gen_result_sc, _ = dp_model(fc_feats, att_feats, att_masks, opt={'sample_max':1}, mode='sample')
                reward = reward_fun(gen_result, fc_feats, true_model).unsqueeze(1).repeat(1, sample_logprobs.size(1))
                reward_sc = reward_fun(gen_result_sc, fc_feats, true_model).unsqueeze(1).repeat(1, sample_logprobs.size(1))
                reward = reward - reward_sc
                loss = rl_crit(sample_logprobs, gen_result.data, reward)
                reward = np.zeros([2, 2])
            elif opt.rl_type == 'reinforce':
                gen_result, sample_logprobs = dp_model(fc_feats, att_feats, att_masks, opt={'sample_max':0}, mode='sample')
                reward = reward_fun(gen_result, fc_feats, true_model).unsqueeze(1).repeat(1, sample_logprobs.size(1))
                loss = rl_crit(sample_logprobs, gen_result.data, reward)
                reward = np.zeros([2, 2])
            elif opt.rl_type == 'reinforce_demean':
                gen_result, sample_logprobs = dp_model(fc_feats, att_feats, att_masks, opt={'sample_max':0}, mode='sample')
                reward = reward_fun(gen_result, fc_feats, true_model).unsqueeze(1).repeat(1, sample_logprobs.size(1))
                loss = rl_crit(sample_logprobs, gen_result.data, reward-reward.mean())
                reward = np.zeros([2, 2])
            elif opt.rl_type == 'arsm':
                loss = get_arm_loss(dp_model, fc_feats, att_feats, att_masks, true_model, opt)
                #print(loss)
                reward = np.zeros([2,2])
            elif opt.rl_type == 'ars':
                loss = get_arm_loss(dp_model, fc_feats, att_feats, att_masks, true_model, opt, type='ars')
                #print(loss)
                reward = np.zeros([2,2])
            elif opt.rl_type == 'ar':
                loss = get_ar_loss(dp_model, fc_feats, att_feats, att_masks, true_model, opt)
                # print(loss)
                reward = np.zeros([2, 2])
            elif opt.rl_type =='mct_baseline':
                opt.rf_demean = 0
                gen_result, sample_logprobs, probs, mct_baseline = get_mct_loss(dp_model, fc_feats, att_feats, att_masks,
                                                                         opt, true_model)
                reward = reward_fun(gen_result, fc_feats, true_model).unsqueeze(1).repeat(1, sample_logprobs.size(1))
                reward_cuda = reward
                #mct_baseline[mct_baseline < 0] = reward_cuda[mct_baseline < 0]
                loss = rl_crit(sample_logprobs, gen_result.data, reward - mct_baseline)
        if opt.mle_weights != 0:
            loss += opt.mle_weights * crit(dp_model(fc_feats, att_feats, labels, att_masks), labels[:, 1:], masks[:, 1:])
        #TODO make sure all sampling replaced by greedy for critic
        #### update the actor
        loss.backward()
        # with open(os.path.join(opt.checkpoint_path, 'best_embed.pkl'), 'wb') as f:
        #     cPickle.dump(list(dp_model.embed.parameters())[0].data.cpu().numpy(), f)
        # with open(os.path.join(opt.checkpoint_path, 'best_logit.pkl'), 'wb') as f:
        #     cPickle.dump(list(dp_model.logit.parameters())[0].data.cpu().numpy(), f)
        ## compute variance
        gradient = torch.zeros([0]).cuda()
        for i in model.parameters():
            gradient = torch.cat((gradient, i.grad.view(-1)), 0)
        first_order = 0.9999 * first_order + 0.0001 * gradient
        second_order = 0.9999 * second_order + 0.0001 * gradient.pow(2)
        # print(torch.max(torch.abs(gradient)))
        variance = torch.mean(torch.abs(second_order - first_order.pow(2))).item()
        if opt.rl_type != 'arsm' or not sc_flag:
            utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        train_loss = loss.item()
        torch.cuda.synchronize()
        end = time.time()
        if (iteration % opt.losses_log_every == 0):
            if not sc_flag:
                print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, train_loss, end - start))
                print(opt.checkpoint_path)
            else:
                print("iter {} (epoch {}), avg_reward = {:.3f}, variance = {:g}, time/batch = {:.3f}" \
                      .format(iteration, epoch, reward.mean(), variance, end - start))

        # Update the iteration and epoch
        iteration += 1
        if (iteration*opt.batch_size) % data_num == 0:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
            add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
            add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
            if sc_flag:
                add_summary_value(tb_summary_writer, 'avg_reward', reward.mean(), iteration)
                add_summary_value(tb_summary_writer, 'variance', variance, iteration)

            #loss_history[iteration] = train_loss if not sc_flag else reward.mean()
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob
            variance_history[iteration] = variance
            time_history[iteration] = end - start


        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):
            # eval model

            val_loss, lang_stats = eval_utils_syn(dp_model, true_model, test_data_features, opt.batch_size, crit)

            lang_stats = lang_stats.item()
            val_loss = val_loss.item()
            # Write validation result into summary
            add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats}
            # Save model if is improving on validation result
            print('loss', val_loss, 'lang_stats', lang_stats)
            if True: # if true
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
                if not os.path.isdir(opt.checkpoint_path):
                    os.mkdir(opt.checkpoint_path)
                torch.save(model.state_dict(), checkpoint_path)
                checkpoint_path = os.path.join(opt.checkpoint_path, 'truemodel.pth')
                torch.save(true_model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
                torch.save(optimizer.state_dict(), optimizer_path)
                # Dump miscalleous informations
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['vocab'] = opt.vocab_size
                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['critic_loss_history'] = critic_loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history
                histories['variance_history'] = variance_history
                histories['time'] = time_history
                # histories['variance'] = 0
                with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(histories, f)

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

opt = opts.parse_opt()
train(opt)
