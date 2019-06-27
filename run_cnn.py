#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import codecs
import os
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as kr
from sklearn import metrics

from cnn_config import TCNNConfig
from cnn_model import TextCNN
from data.data_loader import read_vocab, batch_iter, process_file, build_vocab
from train_word2vec import train_word2vec


def init_vocab(config):
    if not os.path.exists(config.vocab_dir):  # 如果不存在词汇表，重建
        print('build vocabulary')
        build_vocab(config.train_dir, config.vocab_dir, config.vocab_size)
    config.words, config.word_to_id = read_vocab(config.vocab_dir)
    config.vocab_size = len(config.words)


def init_embeddings(config):
    # trans vector file to numpy file
    if not os.path.exists(config.vector_word_npz):
        print('将word2vec的结果转化成numpy格式保存')
        export_word2vec_vectors(config)
    with np.load(config.vector_word_npz) as data:
        config.pre_training = data["embeddings"]


def export_word2vec_vectors(config):
    """
    save vocab_vector to numpy file
    :param config: config
    :return:
    """
    vocab = config.word_to_id
    file_r = codecs.open(config.vector_word_filename, 'r', encoding='utf-8')
    line = file_r.readline()
    voc_size, vec_dim = map(int, line.split(' '))
    embeddings = np.zeros([len(vocab), vec_dim])
    line = file_r.readline()
    while line:
        items = line.split(' ')
        word = items[0]
        vec = np.asarray(items[1:], dtype='float32')
        if word in vocab:
            word_idx = vocab[word]
            embeddings[word_idx] = np.asarray(vec)
        line = file_r.readline()
    np.savez_compressed(config.vector_word_npz, embeddings=embeddings)


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(model, x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, model, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(model, x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train(config, model):
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(config.tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    x_train, y_train = process_file(config.train_dir, config.word_to_id, config.cat_to_id, config.seq_length)
    x_val, y_val = process_file(config.val_dir, config.word_to_id, config.cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(model, x_batch, y_batch, config.dropout_keep_prob)

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, model, x_val, y_val)

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=config.save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


def test(config, model):
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = process_file(config.test_dir, config.word_to_id, config.cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=config.save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, model, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=config.categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def btest(config, model):
    print("Loading test data...")
    contents = ["本院认为,被告马小根欠原告干彩红借款230000元，于本判决生效后十日内还清。"]
    data_id = []
    for i in range(len(contents)):
        data_id.append([config.word_to_id[x] for x in contents[i] if x in config.word_to_id])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_test = kr.preprocessing.sequence.pad_sequences(data_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=config.save_path)  # 读取保存的模型

    print('Testing...')

    feed_dict = {
        model.input_x: x_test[0:1],
        model.keep_prob: 1.0
    }
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    y_pred_cls[0:1] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    categories = ['当事人', '审理经过', '一审原告诉称', '一审被告辩称', '本院查明', '本院认为', '裁判结果', '公诉机关称', '第三人称', '反诉情况', '证据', '一审第三人称',
                  '一审法院查明', '一审法院认为', '上诉人诉称', '被上诉人辩称', '一审公诉机关称']
    id_to_cat = dict(zip(range(len(categories)), categories))
    print(y_pred_cls[0:1])
    print(id_to_cat[y_pred_cls[0:1].item()])


def init(data_path, category_type):
    """
    :param data_path: 数据集路径
    :param category_type: 1一审 2二审
    :return:
    """
    print('Configuring CNN model...')
    config = TCNNConfig(data_path, category_type)
    train_word2vec(config)

    init_vocab(config)
    init_embeddings(config)

    model = TextCNN(config)
    return config, model
