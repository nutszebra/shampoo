import argparse
import math
from models import wide_resnet
from utility_pytorch.trainer_cifar10 import Cifar10Trainer
from utility_pytorch.trainer_cifar100 import Cifar100Trainer
from utility_pytorch.optimizers import MomentumSGD, AdamW, Shampoo
import utility_pytorch.transformers as transformers

parser = argparse.ArgumentParser(description='PyTorch cifar10 Example')
parser.add_argument('--gpu', type=int, default=-1, metavar='N',
                    help='-1 means cpu, otherwise gpu id')
parser.add_argument('--save_path', type=str, default='./log', metavar='N',
                    help='log and model will be saved here')
parser.add_argument('--load_model', default=None, metavar='N',
                    help='pretrained model')
parser.add_argument('--train_batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='epochs start from this number')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--model', type=str, default='squeezenet_pytorch.SqueezeNet(10,1.0e5,1.0e3)', metavar='M',
                    help='model definition here')
parser.add_argument('--optimizer', type=str, default='AdamW(model,lr=1.0e-3,betas=(0.9,0.999),eps=1.0e-8,weight_decay=1.0e-4,schedule=[100,150],lr_decay=0.1)', metavar='M',
                    help='optimizer definition here')
parser.add_argument('--trainer', type=str, default='Cifar10Trainer', metavar='M',
                    help='model definition here')
parser.add_argument('--train_transform', type=str, default=None, metavar='M',
                    help='train transform')
parser.add_argument('--test_transform', type=str, default=None, metavar='M',
                    help='train transform')
args = parser.parse_args().__dict__
print('Args')
print('    {}'.format(args))
model, optimizer, trainer = args.pop('model'), args.pop('optimizer'), args.pop('trainer')
exec('{}={}'.format("args['train_transform']", args['train_transform']))
exec('{}={}'.format("args['test_transform']", args['test_transform']))

# define model
exec('model = {}'.format(model))
print('Model')
print('    name: {}'.format(model.name))
print('    parameters: {}'.format(model.count_parameters()))
# deine optimizer
exec('optimizer = {}'.format(optimizer))
optimizer.info()

args['model'], args['optimizer'] = model, optimizer
exec('main = {}(**args)'.format(trainer))
main.run()
