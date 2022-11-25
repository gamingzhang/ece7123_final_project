import torch
import logging
from datetime import datetime
from torchvision.datasets import CIFAR10,FashionMNIST,MNIST
from initialize import *

def pre_data(data):

	if data == 'CIFAR10':
		
		data_tf = transforms.Compose([
						transforms.ToTensor(),
    					transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    					])

		''' train data'''
		train_set = CIFAR10('./data', train=True, transform=data_tf, download=True)
		train_data = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

		''' test data '''

		test_set = CIFAR10('./data', train=False, transform=data_tf, download=True)
		test_data = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

		return train_data,test_data

	elif data == 'MNIST':
		pass


def check_gpu(logger):
	''' 检查gpu '''
	if torch.cuda.is_available():
		try:			
			gpu_number = int(sys.argv[1])
			torch.cuda.set_device(gpu_number)
		except IndexError as e:
			logger.info('GPU unspecified!')
		logger.info('GPU DEVICE : %s'%torch.cuda.get_device_name(gpu_number))
	else:
   		logger.info('GPU IS UNAVAILABLE!')

def main():
	''' config of logging '''
	logging.basicConfig(level=logging.DEBUG,
                    # filename='CIFAR10.txt',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(levelname)s - %(message)s')
	logger = logging.getLogger(__name__)
	check_gpu(logger)

	program_config = {
		'data_set':'CIFAR10',
		'population_size':20,
		'generation':10,
		'init_epochs':5,
		'final_epochs':10
	}

	individual_config = {
		'part1_max':10,
		'part2_max':3,
		'maps_max':256,
		'neurons_max':300,
		'class_number':10
	}

	start_time = datetime.now()
	logger.info('----------Begin----------')

	p = population(program_config['population_size'],individual_config)




if __name__ == '__main__':
	main()