import os


class TCNNConfig(object):
    """CNN配置参数"""
    embedding_dim = 200  # 词向量维度
    seq_length = 600  # 序列长度
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 50000  # 词汇表达小
    pre_training = None  # use vector_char trained by word2vec

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

    def __init__(self, data_base_dir, category_type):
        self.train_dir = os.path.join(data_base_dir, 'train.txt')
        self.test_dir = os.path.join(data_base_dir, 'test.txt')
        self.val_dir = os.path.join(data_base_dir, 'val.txt')
        self.vocab_dir = os.path.join(data_base_dir, 'vocab.txt')
        self.vector_word_npz = os.path.join(data_base_dir, 'vector_word.npz')
        self.vector_word_filename = os.path.join(data_base_dir, 'vector_word.txt')
        if category_type == 1:
            self.categories = ['当事人', '审理经过', '原告诉称', '被告辩称', '本院查明', '本院认为', '裁判结果', '公诉机关称', '第三人称', '反诉情况', '证据']
            self.save_dir = 'checkpoints/yishen/textcnnative_content(content)n'
            self.tensorboard_dir = 'tensorboard/yishen'
        elif category_type == 2:
            self.categories = ['当事人', '审理经过', '一审原告诉称', '一审被告辩称', '本院查明', '本院认为', '裁判结果', '公诉机关称', '第三人称', '反诉情况', '证据',
                               '一审第三人称', '一审法院查明', '一审法院认为', '上诉人诉称', '被上诉人辩称', '一审公诉机关称']
            self.save_dir = 'checkpoints/ershen/textcnnative_content(content)n'
            self.tensorboard_dir = 'tensorboard/ershen'

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.tensorboard_dir):
            os.makedirs(self.tensorboard_dir)
        self.save_path = os.path.join(self.save_dir, 'best_validation')  # 最佳验证结果保存路径
        self.num_classes = len(self.categories)  # 类别数
        self.cat_to_id = dict(zip(self.categories, range(self.num_classes)))
        self.words = None
        self.word_to_id = None
