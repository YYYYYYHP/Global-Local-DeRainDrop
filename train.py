import os

os.sys.path.append('..')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from options import Options
from train_module import Trainer


if __name__ == '__main__':

    opt = Options().parse()
    trainer = Trainer(opt)
    trainer.train()
    trainer.test()
