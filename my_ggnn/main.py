import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim

from model import GGNN
from utils.data.dataset import bAbIDataset
from utils.data.dataloader import bAbIDataloader

parser = argparse.ArgumentParser()
# question settings
parser.add_argument('--task_id', type=int, default=4, help='bAbI task id')
parser.add_argument('--question_id', type=int, default=0, help='question types')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)

# model settings
parser.add_argument('--annotation_dim', type=int, default=1, help='dim for annotation')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--state_dim', type=int, default=4, help='GGNN hidden state size')
parser.add_argument('--n_steps', type=int, default=5, help='propogation steps number of GGNN')

# training settings
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--verbal', type=bool, default=True, help='print training info or not')
parser.add_argument('--manual_seed', type=int, help='manual seed')
opt = parser.parse_args()

if opt.manual_seed is None:
    opt.manual_seed = random.randint(1, 10000)
random.seed(opt.manual_seed)
torch.manual_seed(opt.manual_seed)

opt.dataroot = 'babi_data/processed_1/train/%d_graphs.txt' % opt.task_id

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manual_seed)




def main(opt):
    train_dataset = bAbIDataset(opt.dataroot, opt.question_id, True)
    train_dataloader = bAbIDataloader(train_dataset, batch_size=opt.batch_size,
                                      shuffle=True, num_workers=2)

    test_dataset = bAbIDataset(opt.dataroot, opt.question_id, False)
    test_dataloader = bAbIDataloader(test_dataset, batch_size=opt.batch_size,
                                     shuffle=False, num_workers=2)


    opt.edge_type_num = train_dataset.edge_type_num
    opt.node_num = train_dataset.node_num

    net = GGNN(opt)
    net.double()

    criterion = nn.CrossEntropyLoss()

    if opt.cuda:
        net.cuda()
        criterion.cuda()

    optimizer = optim.Adam(net.parameters(), lr=opt.lr)

    for epoch in range(0, opt.niter):
        net._train(epoch, train_dataloader, net, criterion, optimizer, opt)
        net._test(test_dataloader, net, criterion, optimizer, opt)




if __name__ == "__main__":
    main(opt)
