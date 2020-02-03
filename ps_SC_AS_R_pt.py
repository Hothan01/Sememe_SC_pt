import sys
sys.path.append('..')
import torch
import random
import os
import utils_pt


if __name__ == '__main__':

    # ################ Prepare Data ###################
    # basic config
    dim = 200                                   # dimension of embedding
    lr_decay_rate = 0.99                        # learning rate decay rate
    batch_size = 1                              # batch size, set to 1 because we use SGD
    learning_rate = 0.01                        # initial learning rate
    # learning_rate  = float(sys.argv[1])
    labda = 0.0001                              # regularization term labmda
    # labda = float(sys.argv[2])                  # regularization term labmda
    total_epoch = 50                            # total training epoches
    # total_epoch = int(sys.argv[3])              # total training epoches
    trunc_num = 5                               # dimention for decomposed sparse matrix
    # trunc_num = int(sys.argv[4])
    hownet_filename = 'dataset/hownet.txt'
    comp_filename = 'dataset/all.bin'
    train_filename = 'dataset/train.bin'
    test_filename = 'dataset/test.bin'
    dev_filename = 'dataset/dev.bin'
    embedding_filename = 'dataset/word_embedding.txt'
    sem_embed_filename = 'dataset/sememe_vector.txt'
    logdir_name = 'phrase_sim/SCASpos_'+'trunc'+str(trunc_num)

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

    U = torch.normal(0, 0.5, (4, 2 * dim, trunc_num), requires_grad=True).float()
    V = torch.normal(0, 0.5, (4, dim, trunc_num), requires_grad=True).float()
    W_c_base = torch.normal(0, 1, (2 * dim, dim), requires_grad=True).float()
    b_c = torch.zeros(1, dim, requires_grad=True).float()

    # regularizer
    '''
    正则化暂不考虑
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(labda)(U))
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(labda)(V))
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(labda)(W_c_base))
    
    '''

    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD([U, V, W_c_base, b_c], lr=learning_rate)

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
            U_i = torch.reshape(U[batch_dict['pos']], (2 * dim, trunc_num))
            V_i = torch.reshape(V[batch_dict['pos']], (dim, trunc_num))
            W_c_i = torch.matmul(U_i, V_i.t())

            embed_aggre_sememe_l = torch.sum(embed_sememe_l, dim=0, keepdim=True)
            embed_aggre_sememe_r = torch.sum(embed_sememe_r, dim=0, keepdim=True)
            embed_word_whole = embed_word_r + embed_word_l
            embed_sememe_whole = embed_aggre_sememe_r + embed_aggre_sememe_l

            phrase_vec = torch.tanh(torch.matmul(torch.cat((embed_word_whole, embed_sememe_whole), 1), W_c_i + W_c_base) + b_c)
            loss = criterion(phrase_vec, embed_truth)
            loss_this_epoch += loss

            if current_num % 100 == 0:
                sys.stdout.flush()
                sys.stdout.write('\rTraining num: ' + str(current_num) + ' of ' + str(train_num) + ' loss:' + str(loss_this_epoch / (0.1 + current_num)))

            loss.backward()
            optimizer.step()

        with open(print_writer_filename, 'a', encoding='utf-8') as fprint:
            fprint.write('epoch: '+str(epoch+1)+' loss:'+str(loss_this_epoch/(0.1+len(hownet.comp_train)))+'\n')

    # dev
    loss_dev = 0
    for current_num, dev_tup in enumerate(hownet.comp_dev):
        batch_dict = utils_pt.generate_one_example(hownet, dev_tup)

        optimizer.zero_grad()

        embed_word_l = word_embedding[batch_dict['wl']].view(1, -1)   # 截取行的数据
        embed_word_r = word_embedding[batch_dict['wr']].view(1, -1)
        embed_truth = word_embedding[batch_dict['lb']].view(1, -1)
        embed_sememe_l = utils_pt.norm(sememe_embedding[batch_dict['sl']])
        embed_sememe_r = utils_pt.norm(sememe_embedding[batch_dict['sr']])
        U_i = torch.reshape(U[batch_dict['pos']], (2 * dim, trunc_num))
        V_i = torch.reshape(V[batch_dict['pos']], (dim, trunc_num))
        W_c_i = torch.matmul(U_i, V_i.t())

        embed_aggre_sememe_l = torch.sum(embed_sememe_l, dim=0, keepdim=True)
        embed_aggre_sememe_r = torch.sum(embed_sememe_r, dim=0, keepdim=True)
        embed_word_whole = embed_word_r + embed_word_l
        embed_sememe_whole = embed_aggre_sememe_r + embed_aggre_sememe_l

        phrase_vec = torch.tanh(torch.matmul(torch.cat((embed_word_whole, embed_sememe_whole), 1), W_c_i + W_c_base) + b_c)
        loss = criterion(phrase_vec, embed_truth)
        loss_dev += loss

        loss.backward()
        optimizer.step()

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
            U_i = torch.reshape(U[batch_test['pos']], (2 * dim, trunc_num))
            V_i = torch.reshape(V[batch_test['pos']], (dim, trunc_num))
            W_c_i = torch.matmul(U_i, V_i.t())

            embed_aggre_sememe_l = torch.sum(embed_sememe_l, dim=0, keepdim=True)
            embed_aggre_sememe_r = torch.sum(embed_sememe_r, dim=0, keepdim=True)
            embed_word_whole = embed_word_r + embed_word_l
            embed_sememe_whole = embed_aggre_sememe_r + embed_aggre_sememe_l

            phrase_vector = torch.tanh(torch.matmul(torch.cat((embed_word_whole, embed_sememe_whole), 1), W_c_i + W_c_base) + b_c)

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

