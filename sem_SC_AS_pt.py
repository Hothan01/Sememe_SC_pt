import sys
sys.path.append('..')
import torch
import random
import os
import utils_pt

# 计算多标签的loss，公式采用了tf.nn.weighted_cross_entropy_with_logits的计算公式
def CrossEntropyLoss(inputs, targets, weight):
    res = (-1) * targets * torch.log(torch.sigmoid(inputs)) * weight + (1 - targets) * torch.log(1 - torch.sigmoid(inputs))
    return torch.mean(res)


if __name__ == '__main__':

    # ################ Prepare Data ###################
    # basic config
    dim = 200                                   # dimension of embedding
    lr_decay_rate = 0.99                        # learning rate decay rate
    batch_size = 1                              # batch size, set to 1 because we use SGD
    learning_rate = 0.2                         # initial learning rate  0.2
    total_epoch = 40                            # total training epoches  40
    k = 100                                     # parameter k for weighted_cross_entropy_with_logits

    hownet_filename = 'dataset/hownet.txt'
    comp_filename = 'dataset/all.bin'
    train_filename = 'dataset/train.bin'
    test_filename = 'dataset/test.bin'
    dev_filename = 'dataset/dev.bin'
    embedding_filename = 'dataset/word_embedding.txt'
    sem_embed_filename = 'dataset/sememe_vector.txt'
    logdir_name = 'sememe_prediction/SCAS'

    # load hownet，并把hownet.comp分成test_set和train_set
    hownet = utils_pt.Hownet(hownet_file=hownet_filename, comp_file=comp_filename)
    hownet.build_hownet()
    hownet.token2id()
    hownet.load_split_dataset(train_filename=train_filename, test_filename=test_filename, dev_filename=dev_filename)
    word_embedding_np, hownet = utils_pt.load_word_embedding(embedding_filename, hownet, scale=True)  # load word embedding
    sememe_embedding_np = utils_pt.load_sememe_embedding(sem_embed_filename, hownet, scale=True)  # load sememe embedding
    train_num = len(hownet.comp_train)
    pos_dict, word_remove = utils_pt.load_hownet_pos()
    hownet, cls_dict = utils_pt.divide_data_with_pos(pos_dict, hownet)
    print("number of dataset in training set:{}".format(len(hownet.comp_train)))
    print("number of dataset in test set:{}".format(len(hownet.comp_test)))
    print("number of dataset in dev set:{}".format(len(hownet.comp_dev)))

    if not os.path.exists(logdir_name):
        os.makedirs(logdir_name)
        os.makedirs(os.path.join(logdir_name, 'print_files'))
        os.makedirs(os.path.join(logdir_name, 'model_file'))
        os.makedirs(os.path.join(logdir_name, 'example_files'))
    # ################ Prepare Data ###################

    # ################ Model and Run ###################
    print_writer_filename = logdir_name + '/print_files/print.txt'  # saver for printing
    word_embedding = torch.from_numpy(word_embedding_np).float()   # numpy to tensor
    sememe_embedding = torch.from_numpy(sememe_embedding_np).float()
    print(word_embedding.dtype)
    print(sememe_embedding.dtype)

    W_c = torch.normal(0, 1.0, (2 * dim, dim), requires_grad=True).float()
    b_c = torch.zeros(1, dim, requires_grad=True).float()

    # criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(k).float(), reduction='mean')
    optimizer = torch.optim.SGD([W_c, b_c], lr=learning_rate)

    state = {'optimizer':optimizer.state_dict()}   # 保存优化器的相关参数
    # saver = tf.train.Saver(max_to_keep=3)  # saver for saving model

    # 这四个参数用来判断是否终止训练
    # these 4 params. are used for deciding whether to stop training
    last_map = 0
    last_last_map = 10
    now_map = 100
    jump2test = False

    random.shuffle(hownet.comp_train)
    for epoch in range(total_epoch):
        example_writer_filename = logdir_name + '/example_files/epoch' + str(epoch + 1) + '.txt'  # file for writing examples
        # train process
        maps_train = []
        loss_train = 0
        for current_num, train_tup in enumerate(hownet.comp_train):
            total_num = epoch * train_num + current_num
            batch_dict = utils_pt.generate_one_example4sememe_prediction(hownet, train_tup)

            optimizer.zero_grad()

            embed_word_l = word_embedding[batch_dict['wl']].view(1, -1)   # 截取行的数据
            embed_word_r = word_embedding[batch_dict['wr']].view(1, -1)
            embed_sememe_l = utils_pt.norm(sememe_embedding[batch_dict['sl']])
            embed_sememe_r = utils_pt.norm(sememe_embedding[batch_dict['sr']])
            labels = torch.from_numpy(batch_dict['lb'])

            embed_aggre_sememe_l_pure = torch.sum(embed_sememe_l, dim=0, keepdim=True)
            embed_aggre_sememe_r_pure = torch.sum(embed_sememe_r, dim=0, keepdim=True)
            embed_word_whole = embed_word_r + embed_word_l
            embed_sememe_whole = embed_aggre_sememe_r_pure + embed_aggre_sememe_l_pure

            phrase_vec = torch.tanh(torch.matmul(torch.cat((embed_word_whole, embed_sememe_whole), 1), W_c) + b_c)
            y_hat = torch.matmul(phrase_vec, sememe_embedding.t())
            loss = CrossEntropyLoss(y_hat, labels, k)

            rank = torch.topk(torch.sigmoid(y_hat), k=hownet.sem_num, largest=True, sorted=True)

            map_score = utils_pt.cal_map_one(batch_dict['al'], rank[1])
            maps_train.append(map_score)

            loss_train += loss

            loss.backward()
            optimizer.step()

            if current_num % 100 == 0:
                sys.stdout.flush()
                sys.stdout.write('\rTraining num: ' + str(current_num) + ' of ' + str(train_num) + '.Epoch:' + str(epoch + 1))
                loss_train = 0
        torch.save(state, logdir_name + '/model_file/model_ckpt-' + str(epoch + 1) )
        # saver.save(sess, logdir_name + '/model_file/model_ckpt', global_step=epoch + 1)
        '''
        目前到这一步，该处理下面的saver，需要将pytorch的这方面好好学一下
        '''
        # dev set test
        maps_dev = []
        loss_dev = 0
        for current_num, dev_tup in enumerate(hownet.comp_dev):
            batch_dev = utils_pt.generate_one_example4sememe_prediction(hownet, dev_tup)

            embed_word_l = word_embedding[batch_dev['wl']].view(1, -1)   # 截取行的数据
            embed_word_r = word_embedding[batch_dev['wr']].view(1, -1)
            embed_sememe_l = utils_pt.norm(sememe_embedding[batch_dev['sl']])
            embed_sememe_r = utils_pt.norm(sememe_embedding[batch_dev['sr']])
            labels = torch.from_numpy(batch_dev['lb'])

            embed_aggre_sememe_l_pure = torch.sum(embed_sememe_l, dim=0, keepdim=True)
            embed_aggre_sememe_r_pure = torch.sum(embed_sememe_r, dim=0, keepdim=True)
            embed_word_whole = embed_word_r + embed_word_l
            embed_sememe_whole = embed_aggre_sememe_r_pure + embed_aggre_sememe_l_pure

            phrase_vec = torch.tanh(torch.matmul(torch.cat((embed_word_whole, embed_sememe_whole), 1), W_c) + b_c)
            y_hat = torch.matmul(phrase_vec, sememe_embedding.t())
            loss = CrossEntropyLoss(y_hat, labels, k)

            rank = torch.topk(torch.sigmoid(y_hat), k=hownet.sem_num, largest=True, sorted=True)

            map_score = utils_pt.cal_map_one(batch_dev['al'], rank[1])
            maps_dev.append(map_score)

            loss_dev += loss

        print('Loss(dev )in epoch %d : %f' % (epoch + 1, loss_dev / len(hownet.comp_dev)))
        print('MAP(dev) in epoch %d : %f' % (epoch + 1, sum(maps_dev) / float(len(hownet.comp_dev))))
        print("*************DEV END*************\n")
        # write log file

        with open(print_writer_filename, 'a', encoding='utf-8') as fp:
            fp.write('\nLoss(dev)in epoch %d:\t%f' % (epoch + 1, loss_dev / len(hownet.comp_dev)))
            fp.write('\nMAP(dev ) in epoch %d:\t%f' % (epoch + 1, sum(maps_dev) / float(len(hownet.comp_dev))))
            fp.write("**************DEV END*************\n")

        # 判断终止条件：至少训练了20个epoch后，若在开发集上，连续两次map值上升，则终止；
        # Deciding the stop condition: After training at least 20 epochs,
        # if the MAP value rises twice in the development set, it stops;
        if epoch+1 == 20:
            last_map = sum(maps_dev) / float(len(hownet.comp_dev))
        if epoch+1 == 21:
            last_last_map = last_map
            last_map = sum(maps_dev) / float(len(hownet.comp_dev))
        elif epoch+1 >= 22:
            now_map = sum(maps_dev) / float(len(hownet.comp_dev))
            if now_map <= last_map <= last_last_map:
                jump2test = True
            else:
                last_last_map = last_map
                last_map = now_map
        if epoch+1 >= 40:
            jump2test = True

        if jump2test:
            model_file = os.path.join(logdir_name, 'model_file')
            if not os.path.exists(model_file):
                print("WARNING: path doesn't exist!")
                sys.exit(0)
            files = os.listdir(model_file)
            '''
            third_last = 'model_ckpt-99'
            for _model in files:
                if _model != 'checkpoint':
                    _model = _model[:13]
                    if _model < third_last:  # 加载倒数第二次map最大的那个文件
                        third_last = _model
            epoch = int(third_last[11:13]) - 1
            '''
            third_last = 'model_ckpt-' + str(epoch - 1)
            meta_file = os.path.join(model_file, third_last + '.meta')
            data_file = os.path.join(model_file, third_last + '.data-00000-of-00001')
            phrase_vec_file = os.path.join(logdir_name, 'example_files', 'phrase_vector.txt')
            third_last = os.path.join(model_file, third_last)

            checkpoint = torch.load(third_last)
            optimizer.load_state_dict(checkpoint['optimizer'])
            # saver.restore(sess, third_last)

            # train set test
            maps_test = []
            loss_test = 0
            for current_num, train_tup in enumerate(hownet.comp_train):
                batch_train = utils_pt.generate_one_example4sememe_prediction(hownet, train_tup)

                embed_word_l = word_embedding[batch_train['wl']].view(1, -1)   # 截取行的数据
                embed_word_r = word_embedding[batch_train['wr']].view(1, -1)
                embed_sememe_l = utils_pt.norm(sememe_embedding[batch_train['sl']])
                embed_sememe_r = utils_pt.norm(sememe_embedding[batch_train['sr']])
                labels = torch.from_numpy(batch_train['lb'])

                embed_aggre_sememe_l_pure = torch.sum(embed_sememe_l, dim=0, keepdim=True)
                embed_aggre_sememe_r_pure = torch.sum(embed_sememe_r, dim=0, keepdim=True)
                embed_word_whole = embed_word_r + embed_word_l
                embed_sememe_whole = embed_aggre_sememe_r_pure + embed_aggre_sememe_l_pure

                phrase_vec = torch.tanh(torch.matmul(torch.cat((embed_word_whole, embed_sememe_whole), 1), W_c) + b_c)
                y_hat = torch.matmul(phrase_vec, sememe_embedding.t())
                loss = CrossEntropyLoss(y_hat, labels, k)

                rank = torch.topk(torch.sigmoid(y_hat), k=hownet.sem_num, largest=True, sorted=True)

                map_score = utils_pt.cal_map_one(batch_train['al'], rank[1])
                maps_test.append(map_score)

                loss_test += loss

            print("************TRAIN START*************")
            print('Loss(train )in epoch %d : %f' % (epoch + 1, loss_test / len(hownet.comp_train)))
            print('MAP(train) in epoch %d : %f' % (epoch + 1, sum(maps_test) / float(len(hownet.comp_train))))
            print("************TRAIN END***************\n")
            # write log file
            with open(print_writer_filename, 'a', encoding='utf-8') as fp:
                fp.write("\n************TRAIN START*************\n")
                fp.write('Loss(train)in epoch %d : %f' % (epoch+1, loss_test/len(hownet.comp_train)))
                fp.write('\nMAP(train) in epoch %d : %f'%(epoch+1,sum(maps_test)/float(len(hownet.comp_train))))
                fp.write("************TRAIN END***************\n")

            # test set test
            maps_test = []
            loss_test = 0
            for current_num, test_tup in enumerate(hownet.comp_test):
                batch_test = utils_pt.generate_one_example4sememe_prediction(hownet, test_tup)

                embed_word_l = word_embedding[batch_test['wl']].view(1, -1)   # 截取行的数据
                embed_word_r = word_embedding[batch_test['wr']].view(1, -1)
                embed_sememe_l = utils_pt.norm(sememe_embedding[batch_test['sl']])
                embed_sememe_r = utils_pt.norm(sememe_embedding[batch_test['sr']])
                labels = torch.from_numpy(batch_test['lb'])

                embed_aggre_sememe_l_pure = torch.sum(embed_sememe_l, dim=0, keepdim=True)
                embed_aggre_sememe_r_pure = torch.sum(embed_sememe_r, dim=0, keepdim=True)
                embed_word_whole = embed_word_r + embed_word_l
                embed_sememe_whole = embed_aggre_sememe_r_pure + embed_aggre_sememe_l_pure

                phrase_vec = torch.tanh(torch.matmul(torch.cat((embed_word_whole, embed_sememe_whole), 1), W_c) + b_c)
                y_hat = torch.matmul(phrase_vec, sememe_embedding.t())
                loss = CrossEntropyLoss(y_hat, labels, k)

                rank = torch.topk(torch.sigmoid(y_hat), k=hownet.sem_num, largest=True, sorted=True)

                map_score = utils_pt.cal_map_one(batch_test['al'], rank[1])
                maps_test.append(map_score)
                _, test_predict = utils_pt.hamming_loss(batch_test['al'], rank[1], get_answer=True, predict_num=hownet.sem_num)
                loss_test += loss

                if len(test_predict) != 0:
                    test_predict_str = utils_pt.predictlabel2char(hownet.id2sememe, test_predict)
                    with open(example_writer_filename, 'a', encoding='utf-8') as ex:
                        ex.write(test_tup[4] + '\n\t')
                        for s in test_predict_str['truth']:
                            ex.write(s + ' ')
                        ex.write('\n\t')
                        for s in test_predict_str['predict']:
                            ex.write(s + ' ')
                        ex.write('\n')

            print("************TEST START*************")
            print('Loss(test )in epoch %d : %f'%(epoch+1,loss_test/len(hownet.comp_test)))
            print('MAP(test) in epoch %d : %f'%(epoch+1,sum(maps_test)/float(len(hownet.comp_test))))
            print("************TEST END***************\n")
            # write log file
            with open(print_writer_filename, 'a', encoding='utf-8') as fp:
                fp.write("\n************TEST START*************\n")
                fp.write('Loss(test )in epoch %d : %f'%(epoch+1,loss_test/len(hownet.comp_test)))
                fp.write('\nMAP(test) in epoch %d : %f'%(epoch+1,sum(maps_test)/float(len(hownet.comp_test))))
                fp.write("************TEST END***************\n")
            break


