from __future__ import print_function
import argparse
import os
from datetime import datetime
from util.logger import setlogger
from util.train_utils_combines import train_utils
import warnings
warnings.filterwarnings('ignore')
import logging

parser = argparse.ArgumentParser(description='UGDBAN')
parser.add_argument('--model_name', type=str, default='NetworkModel',help='Name of the model (in ./models directory)')
parser.add_argument('--signal_size', type=int, default=1024,help='Signal length split by sliding window')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--max_epoch', type=int, default=300, help='max number of epoch')
parser.add_argument('--data_name', type=str, default='', help='the name of the data')
parser.add_argument('--data_dir', type=str, default='',help='the directory of the data')
parser.add_argument('--transfer_task', type=list, default='', help='transfer learning tasks')
parser.add_argument('--normlizetype', type=str, default='', help='nomalization type')
parser.add_argument('--cuda_device', type=str, default='1', help='assign device')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')
parser.add_argument('--no-cuda', action='store_true', default=True, help='disables CUDA training')
parser.add_argument('--seed', type=int, default='', metavar='S', help='random seed (default: 1)')
parser.add_argument("--pretrained", type=bool, default=False, help='whether to load the pretrained model')
parser.add_argument('--num_k', type=int, default='', metavar='K', help='how many steps to repeat the generator update')
parser.add_argument('--num_layer', type=int, default='', metavar='K', help='how many layers for classifier')
parser.add_argument('--class_num', type=int, default='', metavar='B', help='The number of classes')
parser.add_argument('--middle_num', type=int, default='', help='The number of classes')
parser.add_argument('--save_dir', type=str, default='./ckpt',help='Directory to save logs and model checkpoints')
# optimization information
parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='', help='the optimizer')
parser.add_argument('--lr', type=float, default=0.001, help='the initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='Betas for adam')
parser.add_argument('--weightdecay', type=float, default=5e-4, help='Weight decay for both sgd and adam')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloader')
parser.add_argument('--save', type=str, default='./pt', help='Save logs and trained model checkpoints')
parser.add_argument('--load_path', type=str, default='',help='Load trained model checkpoints from this path (for testing, not for resuming training)')
########pseudo-labels
parser.add_argument('--no-progress', action='store_true',help="don't use progress bar")
parser.add_argument('--uncertainty', type=bool, default=True, help='whether to load the pretrained model')
parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
parser.add_argument('--sel_ratio', type=float, default='', help='sel_ratio for clean_samples')
parser.add_argument('--knn_times', type=int, default=2, help='how many times of knn is conducted')
parser.add_argument('--balance_class', type=bool, default=True,help='whether to balance class in pair_selection')

if __name__ == "__main__":
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Prepare the saving path for the model
    sub_dir = args.model_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'train.log'))
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))
    trainer = train_utils(args, save_dir)
    trainer.setup()
    best_acc = 0.0
    for t in range(args.max_epoch):
        trainer.train(t)
        new_acc=trainer.test(t,best_acc)
        if new_acc >= best_acc:
            best_acc = new_acc
            best_epoch = t
        print('The best model epoch {},val_acc {:.6f}'.format(best_epoch, best_acc))

