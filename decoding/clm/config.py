class Config(object):

    def __init__(self):

        self.mode = 'tied'  # 'base', 'tied'
        if self.mode == 'tied':
            self.tied = True
        elif self.mode == 'base':
            self.tied = False
        else:
            raise ValueError('Not acceptable mode')

        self.torch_seed = 87654

        self.wsj_path = '/home/khshim/data/wsj/'
        self.save_path = './best_model_' + self.mode + '.pt'
        self.batch_size = 64
        self.sequence_length = 128
        self.eval_batch_size = 64

        self.max_epoch = 1000
        self.max_change = 4
        self.max_patience = 5

        self.initial_lr = 2e-3
        self.momentum = 0.9
        self.clip_norm = 1


