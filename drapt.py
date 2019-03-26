opt.input_json='data/cocotalk.json'
opt.input_fc_dir='/work/06008/xf993/maverick2/IC/data/cocobu_fc'
opt.input_att_dir='work/06008/xf993/maverick2/IC/data/cocobu_att'
opt.input_label_h5='data/cocotalk_label.h5'

# Deal with feature things before anything
opt.use_att = utils.if_use_att(opt.caption_model)
if opt.use_box: opt.att_feat_size = opt.att_feat_size + 5

loader = DataLoader(opt)
opt.vocab_size = loader.vocab_size
opt.seq_length = loader.seq_length

tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)

infos = {}
histories = {}
if opt.start_from is not None:
    # open old infos and check if models are compatible
    with open(os.path.join(opt.start_from, 'infos_' + opt.id + '.pkl')) as f:
        infos = cPickle.load(f)
        saved_model_opt = infos['opt']
        need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
        for checkme in need_be_same:
            assert vars(saved_model_opt)[checkme] == vars(opt)[
                checkme], "Command line argument and saved model disagree on '%s' " % checkme

    if os.path.isfile(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl')):
        with open(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl')) as f:
            histories = cPickle.load(f)

iteration = infos.get('iter', 0)
epoch = infos.get('epoch', 0)

val_result_history = histories.get('val_result_history', {})
loss_history = histories.get('loss_history', {})
lr_history = histories.get('lr_history', {})
ss_prob_history = histories.get('ss_prob_history', {})

loader.iterators = infos.get('iterators', loader.iterators)
loader.split_ix = infos.get('split_ix', loader.split_ix)
if opt.load_best_score == 1:
    best_val_score = infos.get('best_val_score', None)

model = models.setup(opt).cuda()
dp_model = torch.nn.DataParallel(model)

update_lr_flag = True
# Assure in training mode
dp_model.train()

crit = utils.LanguageModelCriterion()
rl_crit = utils.RewardCriterion()

optimizer = utils.build_optimizer(model.parameters(), opt)
# Load the optimizer
if vars(opt).get('start_from', None) is not None and os.path.isfile(os.path.join(opt.start_from, "optimizer.pth")):
    optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

while True:
if update_lr_flag:
    # Assign the learning rate
    if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
        frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
        decay_factor = opt.learning_rate_decay_rate ** frac
        opt.current_lr = opt.learning_rate * decay_factor
    else:
        opt.current_lr = opt.learning_rate
    utils.set_lr(optimizer, opt.current_lr)
    # Assign the scheduled sampling prob
    if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
        frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
        opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
        model.ss_prob = opt.ss_prob

    # If start self critical training
    if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
        sc_flag = True
        init_scorer(opt.cached_tokens)
    else:
        sc_flag = False

    update_lr_flag = False

start = time.time()
# Load data from train split (0)
data = loader.get_batch('train')
print('Read data:', time.time() - start)

torch.cuda.synchronize()
start = time.time()

tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
fc_feats, att_feats, labels, masks, att_masks = tmp

optimizer.zero_grad()
loss = crit(dp_model(fc_feats, att_feats, labels, att_masks), labels[:, 1:], masks[:, 1:])
loss.backward()
utils.clip_gradient(optimizer, opt.grad_clip)
optimizer.step()
train_loss = loss.item()
torch.cuda.synchronize()
end = time.time()
if not sc_flag:
    print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
          .format(iteration, epoch, train_loss, end - start))
else:
    print("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}" \
          .format(iteration, epoch, np.mean(reward[:, 0]), end - start))

# Update the iteration and epoch
iteration += 1
if data['bounds']['wrapped']:
    epoch += 1
    update_lr_flag = True

# Write the training loss summary
if (iteration % opt.losses_log_every == 0):
    add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
    add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
    add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
    if sc_flag:
        add_summary_value(tb_summary_writer, 'avg_reward', np.mean(reward[:, 0]), iteration)

    loss_history[iteration] = train_loss if not sc_flag else np.mean(reward[:, 0])
    lr_history[iteration] = opt.current_lr
    ss_prob_history[iteration] = model.ss_prob

# make evaluation on validation set, and save model
if (iteration % opt.save_checkpoint_every == 0):
    # eval model
    eval_kwargs = {'split': 'val',
                   'dataset': opt.input_json}
    eval_kwargs.update(vars(opt))
    val_loss, predictions, lang_stats = eval_utils.eval_split(dp_model, crit, loader, eval_kwargs)

    # Write validation result into summary
    add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)
    if lang_stats is not None:
        for k, v in lang_stats.items():
            add_summary_value(tb_summary_writer, k, v, iteration)
    val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

    # Save model if is improving on validation result
    if opt.language_eval == 1:
        current_score = lang_stats['CIDEr']
    else:
        current_score = - val_loss

    best_flag = False
    if True:  # if true
        if best_val_score is None or current_score > best_val_score:
            best_val_score = current_score
            best_flag = True
        checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print("model saved to {}".format(checkpoint_path))
        optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
        torch.save(optimizer.state_dict(), optimizer_path)

        # Dump miscalleous informations
        infos['iter'] = iteration
        infos['epoch'] = epoch
        infos['iterators'] = loader.iterators
        infos['split_ix'] = loader.split_ix
        infos['best_val_score'] = best_val_score
        infos['opt'] = opt
        infos['vocab'] = loader.get_vocab()

        histories['val_result_history'] = val_result_history
        histories['loss_history'] = loss_history
        histories['lr_history'] = lr_history
        histories['ss_prob_history'] = ss_prob_history
        with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + '.pkl'), 'wb') as f:
            cPickle.dump(infos, f)
        with open(os.path.join(opt.checkpoint_path, 'histories_' + opt.id + '.pkl'), 'wb') as f:
            cPickle.dump(histories, f)

        if best_flag:
            checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print("model saved to {}".format(checkpoint_path))
            with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + '-best.pkl'), 'wb') as f:
                cPickle.dump(infos, f)

# Stop if reaching max epochs
if epoch >= opt.max_epochs and opt.max_epochs != -1:
    break






batch_size = fc_feats.size(0)
state = model.init_hidden(batch_size)
outputs = []

for i in range(seq.size(1)):
    if i == 0:
        xt = model.img_embed(fc_feats)
    else:
        if self.training and i >= 2 and self.ss_prob > 0.0: # otherwiste no need to sample
            sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
            sample_mask = sample_prob < self.ss_prob
            if sample_mask.sum() == 0:
                it = seq[:, i-1].clone()
            else:
                sample_ind = sample_mask.nonzero().view(-1)
                it = seq[:, i-1].data.clone()
                #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
        else:
            it = seq[:, i-1].clone()
        # break if all the sequences end
        if i >= 2 and seq[:, i-1].sum() == 0:
            break
        xt = self.embed(it)

    output, state = model.core(xt, state)
    output = F.log_softmax(model.logit(output), dim=1)
    outputs.append(output)




STANFORD_CORENLP_3_4_1_JAR = 'stanford-corenlp-3.4.1.jar'

# punctuations to be removed from the sentences
PUNCTUATIONS = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
        ".", "?", "!", ",", ":", "-", "--", "...", ";"]


cmd = ['java', '-cp', STANFORD_CORENLP_3_4_1_JAR, \
        'edu.stanford.nlp.process.PTBTokenizer', \
        '-preserveLines', '-lowerCase']

# ======================================================
# prepare data for PTB Tokenizer
# ======================================================
final_tokenized_captions_for_image = {}
image_id = [k for k, v in captions_for_image.items() for _ in range(len(v))]
sentences = '\n'.join([c['caption'].replace('\n', ' ') for k, v in captions_for_image.items() for c in v])

# ======================================================
# save sentences to temporary file
# ======================================================
path_to_jar_dirname=os.path.dirname(os.path.abspath(__file__))
tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=path_to_jar_dirname)
tmp_file.write(sentences)
tmp_file.close()

# ======================================================
# tokenize sentence
# ======================================================
cmd.append(os.path.basename(tmp_file.name))
print(path_to_jar_dirname)
p_tokenizer = subprocess.Popen(cmd, cwd=path_to_jar_dirname, \
        stdout=subprocess.PIPE)
token_lines = p_tokenizer.communicate(input=sentences.rstrip())[0]
lines = token_lines.split('\n')
# remove temp file
os.remove(tmp_file.name)

# ======================================================
# create dictionary for tokenized captions
# ======================================================
for k, line in zip(image_id, lines):
    if not k in final_tokenized_captions_for_image:
        final_tokenized_captions_for_image[k] = []
    tokenized_caption = ' '.join([w for w in line.rstrip().split(' ') \
            if w not in PUNCTUATIONS])
    final_tokenized_captions_for_image[k].append(tokenized_caption)

    return final_tokenized_captions_for_image
