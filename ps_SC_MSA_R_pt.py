import sys
sys.path.append('..')
import torch
from torch.nn import functional as F
import random
import os
import utils_pt


def clip_gradient(optimizer, grad_clip):   # 对应源码中优化器中的tf.clip_by_value()
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


if __name__ == '__main__':

    # ################ Prepare Data ###################
    # basic config
    dim = 200                                   # dimension of embedding
    lr_decay_rate = 0.99                        # learning rate decay rate
    batch_size = 1                              # batch size, set to 1 because we use SGD
    learning_rate = 0.01                        # initial learning rate
    labda = 0.0001                              # regularization term labmda
    total_epoch = 50                            # total training epoches
    trunc_num = 5                               # dimention for decomposed sparse matrix
    hownet_filename = 'dataset/hownet.txt'
    comp_filename = 'dataset/all.bin'
    train_filename = 'dataset/train.bin'
    test_filename = 'dataset/test.bin'
    dev_filename = 'dataset/dev.bin'
    embedding_filename = 'dataset/word_embedding.txt'
    sem_embed_filename = 'dataset/sememe_vector.txt'
    logdir_name = 'phrase_sim/SCMSApos_'+'trunc'+str(trunc_num)

    # load hownet，并把hownet.comp分成test_set和train_set
    hownet = utils_pt.Hownet(hownet_file=hownet_filename, comp_file=comp_filename)
    hownet.build_hownet()
    hownet.token2id()
    hownet.load_split_dataset(train_filename=train_filename, test_filename=test_filename, dev_filename=dev_filename)
    word_embedding_np, hownet = utils_pt.load_word_embedding(embedding_filename, hownet, scale=False)  # load word embedding
    sememe_embedding_np = utils_pt.load_sememe_embedding(sem_embed_filename, hownet, scale=True)  # load sememe embedding
    hownet, wordsim_words = utils_pt.fliter_wordsim_all(hownet)  # remove MWEs in testset
    train_num = len(hownet.comp_train)
    pos_dict, word_remove = utils_pt.load_hownet_pos()
    hownet, cls_dict = utils_pt.divide_data_with_pos(pos_dict, hownet)
    print("number of dataset in training set:{}".format(len(hownet.comp_train)))
    print("number of dataset in test set:{}".format(len(hownet.comp_test)))
    print("number of dataset in dev set:{}".format(len(hownet.comp_dev)))

    if not os.path.exists(logdir_name):
        os.makedirs(logdir_name)
        os.makedirs(os.path.join(logdir_name, 'print_files'))
        os.makedirs(os.path.join(logdir_name, 'example_files'))
    # ################ Prepare Data ###################

    # ################ Model and Run ###################

    print_writer_filename = logdir_name + '/print_files/print.txt'  # saver for printing
    word_embedding = torch.from_numpy(word_embedding_np).float()   # numpy to tensor
    sememe_embedding = torch.from_numpy(sememe_embedding_np).float()
    print(word_embedding.dtype)
    print(sememe_embedding.dtype)

    class SCMSAR(torch.nn.Module):
        def __init__(self, dim, trunc_num):
            super(SCMSAR, self).__init__()
            self.linear1 = torch.nn.Linear(dim, dim)   # W_a, b_a
            # 参数初始化
            self.linear1.weight.data = torch.normal(0, 0.5, size=self.linear1.weight.data.size())
            self.linear1.bias.data = torch.zeros(size=self.linear1.bias.data.size())
            # 自定义内部需要更新的参数
            u = torch.normal(0, 0.5, (4, 2 * dim, trunc_num), requires_grad=True)   # torch初始化时，默认dtype:float32
            v = torch.normal(0, 0.5, (4, dim, trunc_num), requires_grad=True)
            w_c_base = torch.normal(0, 1, (2 * dim, dim), requires_grad=True)
            b_c = torch.zeros(1, dim, requires_grad=True)
            self.weight_U = torch.nn.Parameter(u)
            self.weight_V = torch.nn.Parameter(v)
            self.weight_W = torch.nn.Parameter(w_c_base)
            self.bias = torch.nn.Parameter(b_c)

        def forward(self, pos, dim, trunc_num, input_word_l, input_word_r, input_sememe_l, input_sememe_r):
            u_i = torch.reshape(self.weight_U[pos], (2 * dim, trunc_num))
            v_i = torch.reshape(self.weight_V[pos], (dim, trunc_num))
            w_c_i = torch.matmul(u_i, v_i.t())
            # attention
            embed_sememe_l = F.softmax(torch.matmul(input_sememe_l, torch.tanh(self.linear1(input_word_r)).t()), dim=0) * input_sememe_l
            embed_aggre_word_l_pure = torch.sum(embed_sememe_l, dim=0, keepdim=True)
            embed_sememe_r = F.softmax(torch.matmul(input_sememe_r, torch.tanh(self.linear1(input_word_l)).t()), dim=0) * input_sememe_r
            embed_aggre_word_r_pure = torch.sum(embed_sememe_r, dim=0, keepdim=True)
            embed_word_whole = input_word_r + input_word_l
            embed_sememe_whole = embed_aggre_word_r_pure + embed_aggre_word_l_pure
            output = torch.tanh(torch.matmul(torch.cat((embed_word_whole, embed_sememe_whole), 1), w_c_i + self.weight_W) + self.bias)
            return output

    model = SCMSAR(dim, trunc_num)

    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay_rate)   # 学习率衰减

    # training
    random.shuffle(hownet.comp_train)
    for epoch in range(total_epoch):
        loss_this_epoch = 0
        print("Epoch: " + str(epoch + 1))
        for current_num, train_tup in enumerate(hownet.comp_train):
            total_num = epoch * train_num + current_num
            batch_dict = utils_pt.generate_one_example(hownet, train_tup)

            optimizer.zero_grad()

            embed_word_l = word_embedding[batch_dict['wl']].view(1, -1)   # 截取行的数据
            embed_word_r = word_embedding[batch_dict['wr']].view(1, -1)
            embed_truth = word_embedding[batch_dict['lb']].view(1, -1)
            embed_sememe_l = utils_pt.norm(sememe_embedding[batch_dict['sl']])
            embed_sememe_r = utils_pt.norm(sememe_embedding[batch_dict['sr']])
            pos = batch_dict['pos']

            phrase_vec = model(pos, dim, trunc_num, embed_word_l, embed_word_r, embed_sememe_l, embed_sememe_r)
            L2_loss = labda * (torch.norm(model.weight_U, 2) + torch.norm(model.weight_V, 2) + torch.norm(model.linear1.weight, 2) + torch.norm(model.weight_W, 2))
            loss = criterion(phrase_vec, embed_truth) + L2_loss
            loss_this_epoch += loss

            if current_num % 100 == 0:
                sys.stdout.flush()
                sys.stdout.write('\rTraining num: ' + str(current_num) + ' of ' + str(train_num) + ' loss:' + str(loss_this_epoch / (0.1 + current_num)))

            loss.backward()
            clip_gradient(optimizer, 5.0)
            optimizer.step()
        scheduler.step()

        with open(print_writer_filename, 'a', encoding='utf-8') as fprint:
            fprint.write('epoch: '+str(epoch+1)+' loss:'+str(loss_this_epoch/(0.1+len(hownet.comp_train)))+'\n')

    # dev
    loss_dev = 0
    for current_num, dev_tup in enumerate(hownet.comp_dev):
        batch_dict = utils_pt.generate_one_example(hownet, dev_tup)

        embed_word_l = word_embedding[batch_dict['wl']].view(1, -1)   # 截取行的数据
        embed_word_r = word_embedding[batch_dict['wr']].view(1, -1)
        embed_truth = word_embedding[batch_dict['lb']].view(1, -1)
        embed_sememe_l = utils_pt.norm(sememe_embedding[batch_dict['sl']])
        embed_sememe_r = utils_pt.norm(sememe_embedding[batch_dict['sr']])
        pos = batch_dict['pos']

        phrase_vec = model(pos, dim, trunc_num, embed_word_l, embed_word_r, embed_sememe_l, embed_sememe_r)
        L2_loss = labda * (torch.norm(model.weight_U, 2) + torch.norm(model.weight_V, 2) + torch.norm(model.linear1.weight, 2) + torch.norm(model.weight_W, 2))
        loss = criterion(phrase_vec, embed_truth) + L2_loss
        loss_dev += loss

    sys.stdout.flush()
    sys.stdout.write('\nDev set loss:' + str(loss_dev / (0.1 + len(hownet.comp_dev))) + '\n')
    with open(print_writer_filename, 'a', encoding='utf-8') as fprint:
        fprint.write('Dev set loss:' + str(loss_dev / (0.1 + len(hownet.comp_dev))) + '\n')

    # test MWE similarity: write embedding to phrase_vec_file
    number = 0
    phrase_vec_file = os.path.join(logdir_name, 'example_files', 'phrase_vector_epoch.txt')
    for current_num, test_tup in enumerate(hownet.comp_test):
        if test_tup[4] in wordsim_words:
            batch_test = utils_pt.generate_one_example(hownet, test_tup)

            embed_word_l = word_embedding[batch_test['wl']].view(1, -1)   # 截取行的数据
            embed_word_r = word_embedding[batch_test['wr']].view(1, -1)
            embed_truth = word_embedding[batch_test['lb']].view(1, -1)
            embed_sememe_l = utils_pt.norm(sememe_embedding[batch_test['sl']])
            embed_sememe_r = utils_pt.norm(sememe_embedding[batch_test['sr']])
            pos = batch_test['pos']

            phrase_vector = model(pos, dim, trunc_num, embed_word_l, embed_word_r, embed_sememe_l, embed_sememe_r)

            with open(phrase_vec_file, 'a', encoding='utf-8') as f_phrase_embed:
                f_phrase_embed.write(test_tup[4] + ' ')
                phrase_vector = phrase_vector.tolist()[0]
                phrase_vector = [str(vec) for vec in phrase_vector]
                f_phrase_embed.write(' '.join(phrase_vector))
                f_phrase_embed.write('\n')
            number += 1
    print('Have written {} words to phrase_vector.txt'.format(number))
    with open(print_writer_filename, 'a', encoding='utf-8') as fprint:
        fprint.write('Have written {} words to phrase_vector.txt'.format(number))
