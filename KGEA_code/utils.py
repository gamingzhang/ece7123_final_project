import math

import torch.nn.init as init
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import copy
import pickle

from scipy.spatial.distance import cdist
from scipy.special import comb
import itertools
import time

# Fully Convolutional Model
class LeNetFCN(nn.Module):
    def __init__(self):        
        super(LeNetFCN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.conv3 = nn.Conv2d(50, 500, 4, 1)
        self.conv4 = nn.Conv2d(500, 10, 1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        
        x = F.relu(self.conv3(x))
        
        x = self.conv4(x)
        return x.view(-1, 10)
    
    def name(self):
        return "LeNetFCN"


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
  
def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True)

        # if args.half:
        input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
    print(' * Prec@1 {top1.avg:.3f}, loss {loss.avg:.4f}'
          .format(top1=top1, loss=losses))

    return top1.avg, losses.avg
 
def train_data_loader():
	root = '../data'
	download = True
	trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
	train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)

	batch_size = 100    

	#### Sample data begin
	labels = train_set.train_labels.numpy()
	data = train_set.train_data.numpy()
	labels_samples = []
	data_samples = []
	for i in range(0, 10000, 10):
		labels_samples.append(labels[i])
		data_samples.append(data[i,...])

	label_st = torch.from_numpy(np.array(labels_samples))
	data_st = torch.from_numpy(np.array(data_samples))
	train_set.train_data = data_st
	train_set.train_labels = label_st
	#### Sample data end

	train_loader = torch.utils.data.DataLoader(
	             dataset=train_set,
	             batch_size=batch_size,
	             shuffle=True)

	return train_loader

def test_data_loader():
    root = '../data'
    download = True
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    test_set = dset.MNIST(root=root, train=False, transform=trans, download=download)
    
    batch_size = 100
    
    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=batch_size,
                    shuffle=False)
    
    return test_loader

def construct_data_model_criterion(model=None):
	# configure
	class Test():
	    pass
	args = Test()

	args.save_dir = './'
	args.evaluate = 'e'
	args.batch_size = 128
	args.workers = 4
	args.half = True
	args.lr = 0.05
	args.momentum = 0.9
	args.weight_decay = 5e-4
	args.epochs = 300
	args.print_freq = 20
	if model==None:
		# initialize model   
		lenet = LeNetFCN()
		model = lenet

		model.cuda()
		# Load check point    
		checkpoint = torch.load('./TrainedModel.pkl')
		model.load_state_dict(checkpoint)

	# Load data
	val_loader = train_data_loader()

	criterion = nn.CrossEntropyLoss().cuda()
	model.half()
	criterion.half()

	return val_loader, model, criterion

def test_forward(val_loader, model, criterion):  # Validate
	acc, loss = validate(val_loader, model, criterion)
	return acc, loss