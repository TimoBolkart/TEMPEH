# '''
# tensorboard logger

# view in tensorboard
# 1. call the following in command line
# $ tensorboard --logdir=<your_log_dir> --port=<your_port_number, e.g. 6006>

# 2. open link in browsers: localhost:6006

# Please see LICENSE for the licensing information
# '''

from utils.utils import safe_mkdir
from torch.utils.tensorboard import SummaryWriter

# -----------------------------------------------------------------------------

class Logger(object):
    def __init__(self, log_dir=''):
        self.log_dir = log_dir
        safe_mkdir(self.log_dir)
        self.writer = SummaryWriter(log_dir)

    def add_scalar(self, name, data, step):
        self.writer.add_scalar(name, data, step)

    def add_image(self, name, data, step):
        self.writer.add_image(name, data, step)       