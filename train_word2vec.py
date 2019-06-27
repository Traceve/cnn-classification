#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging
import os
import re
import time

from gensim.models import Word2Vec

import jieba

stoplist = {}.fromkeys([line.strip() for line in open("./data/stopwords.txt")])


class Get_Sentences(object):

    def __init__(self, file_names):
        self.filenames = file_names

    def __iter__(self):
        for file_name in self.filenames:
            index = 1
            with open(file_name, 'r') as f:
                for line in f:
                    print(str(index) + ' ' + line[0:20])
                    index += 1
                    doc = line.split('\t')[1]
                    for sentence in doc.split("。"):
                        yield [word for word in (jieba.lcut(re.sub('[\d ×()]', '', sentence))) if word not in stoplist]


def train_word2vec(config):
    if os.path.exists(config.vector_word_filename):  # 如果不存在word2vec文件，则训练word2vec
        return
    print('Train Word2Vec...')
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    t1 = time.time()
    sentences = Get_Sentences([config.train_dir, config.val_dir, config.test_dir])
    model = Word2Vec(sentences, sg=1, hs=1, min_count=1, window=3, size=200, workers=4, iter=2)
    model.wv.save_word2vec_format(config.vector_word_filename, binary=False)
    print('-------------------------------------------')
    print("Training word2vec model cost %.3f seconds...\n" % (time.time() - t1))


if __name__ == '__main__':
    train_word2vec()
