import os
import pickle
import random
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import torch
from tensorboardX import SummaryWriter
from dateutil.relativedelta import   relativedelta

class Logger(object):
    """Reference: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514"""

    def __init__(self, fn, ask=True,minus_1day=False,dir_name="./logs/"):
        self.minus_tag=minus_1day
        self.dir_name=dir_name
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        logdir = self._make_dir(fn)
        print('=============={}=============='.format(logdir))
        if not os.path.exists(logdir):
            os.mkdir(logdir)

        if len(os.listdir(logdir)) != 0 and ask:
            ans='y'
            # ans = input("log_dir is not empty. All data inside log_dir will be deleted. "
            #             "Will you proceed [y/N]? ")
            if ans in ['y', 'Y']:
                shutil.rmtree(logdir)
            else:
                exit(1)

        self.set_dir(logdir)

    def _make_dir(self, fn):
        #
        if self.minus_tag:
            today = datetime.today() - relativedelta(days=1)
            today=today.strftime("%y%m%d")
        else:
            today = datetime.today().strftime("%y%m%d")
        logdir = self.dir_name + today + '_' + fn

        return logdir

    def set_dir(self, logdir, log_fn='log.txt'):
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        self.writer = SummaryWriter(logdir)
        self.log_file = open(os.path.join(logdir, log_fn), 'a')

    def log(self, string):
        self.log_file.write('[%s] %s' % (datetime.now(), string) + '\n')
        self.log_file.flush()

        print('[%s] %s' % (datetime.now(), string))
        sys.stdout.flush()

    def log_dirname(self, string):
        self.log_file.write('%s (%s)' % (string, self.logdir) + '\n')
        self.log_file.flush()

        print('%s (%s)' % (string, self.logdir))
        sys.stdout.flush()

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        self.writer.add_image(tag, images, step)

    def histo_summary(self, tag, values, step):
        """Log a histogram of the tensor of values."""
        self.writer.add_histogram(tag, values, step, bins='auto')

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)