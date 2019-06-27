#!/usr/bin/python
# -*- coding: utf-8 -*-
import run_cnn

if __name__ == '__main__':
    config, model = run_cnn.init('data/ershen/small', 2)
    run_cnn.train(config, model)
