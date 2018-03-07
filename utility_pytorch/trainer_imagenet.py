import torch
import numpy as np
from torchvision import transforms
from torch.autograd import Variable
from .utility import remove_slash, make_dir, create_progressbar, write
from .log import Log
from .pickle_loader import PickleFolder


class ImagenetTrainer(object):

    def __init__(self, model, optimizer, data_path, gpu=-1, save_path='./', load_model=None, train_transform=None, test_transform=None, train_batch_size=64, test_batch_size=256, start_epoch=1, epochs=200, seed=1):
        self.model, self.optimizer, self.data_path = model, optimizer, remove_slash(data_path)
        self.gpu, self.save_path, load_model = gpu, remove_slash(save_path), load_model
        self.train_transform, self.test_transform = train_transform, test_transform
        self.train_batch_size, self.test_batch_size = train_batch_size, test_batch_size
        self.start_epoch, self.epochs, self.seed = start_epoch, epochs, seed
        # load mnist
        self.init_dataset()
        # initialize seed
        self.init_seed()
        # create directory
        make_dir(save_path)
        # load pretrained model if possible
        self.load(load_model)
        # init log
        self.init_log()

    def init_transform(self):
        if self.train_transform is None:
            self.train_transform = transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        if self.test_transform is None:
            self.test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

    def init_dataset(self):
        # initialize transform
        self.init_transform()
        # arguments for gpu mode
        kwargs = {}
        if self.check_gpu():
            kwargs = {'num_workers': 1,
                      'pin_memory': True}

        # load dataset
        train_dataset = PickleFolder('{}/{}'.format(self.data_path, 'train'), transform=self.train_transform)
        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=self.train_batch_size,
                                                        shuffle=True, **kwargs)
        test_dataset = PickleFolder('{}/{}'.format(self.data_path, 'val'), transform=self.test_transform)
        self.test_loader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=self.test_batch_size,
                                                       shuffle=False, **kwargs)

    def init_log(self):
        self.log = {}
        self.log['train_loss'] = Log(self.save_path, 'train_loss.log')
        self.log['test_loss'] = Log(self.save_path, 'test_loss.log')
        self.log['test_accuracy'] = Log(self.save_path, 'test_accuracy.log')
        self.log['test_accuracy'] = Log(self.save_path, 'test_5_accuracy.log')

    def check_gpu(self):
        return self.gpu >= 0 and torch.cuda.is_available()

    def init_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.check_gpu():
            torch.cuda.manual_seed(self.seed)

    def to_gpu(self):
        if self.check_gpu():
            self.model = self.model.cuda(self.gpu)

    def to_cpu(self):
        self.model = self.model.cpu()

    def train_one_epoch(self):
        self.to_gpu()
        self.model.train()
        sum_loss = 0
        progressbar = create_progressbar(self.train_loader, desc='train')
        for x, t in progressbar:
            if self.check_gpu():
                x, t = x.cuda(self.gpu), t.cuda(self.gpu)
            x, t = Variable(x, volatile=False), Variable(t, volatile=False)
            self.optimizer.zero_grad()
            y = self.model(x)
            loss = self.model.calc_loss(y, t)
            loss.backward()
            self.optimizer.step()
            sum_loss += loss.cpu().data[0] * self.train_batch_size
        self.to_cpu()
        return sum_loss / len(self.train_loader.dataset)

    def test_one_epoch(self):
        self.to_gpu()
        self.model.eval()
        sum_loss = 0
        accuracy = 0
        progressbar = create_progressbar(self.test_loader, desc='test')
        for x, t in progressbar:
            if self.check_gpu():
                x, t = x.cuda(self.gpu), t.cuda(self.gpu)
            x, t = Variable(x, volatile=True), Variable(t, volatile=True)
            y = self.model(x)
            # loss
            loss = self.model.calc_loss(y, t)
            sum_loss += loss.cpu().data[0] * self.test_batch_size
            # accuracy
            y = y.data.max(1, keepdim=True)[1]
            accuracy += y.eq(t.data.view_as(y)).cpu().sum()
        sum_loss /= len(self.test_loader.dataset)
        accuracy /= len(self.test_loader.dataset)
        self.to_cpu()
        return sum_loss, accuracy

    def save(self, i):
        self.model.eval()
        torch.save(self.model.state_dict(), '{}/{}_{}.model'.format(self.save_path, self.model.name, i))

    def load(self, path=None):
        if path is not None:
            write('load {}'.format(path))
            self.model.eval()
            self.model.load_state_dict(torch.load(path))
        else:
            write('weight initilization')
            self.model.weight_initialization()

    def run(self):
        for i in create_progressbar(self.epochs + 1, desc='epoch', stride=1, start=self.start_epoch):
            train_loss = self.train_one_epoch()
            self.log['train_loss'].write('{}'.format(train_loss), debug='Loss Train {}:'.format(i))
            self.save(i)
            self.optimizer(i)
            test_loss, test_accuracy = self.test_one_epoch()
            self.log['test_loss'].write('{}'.format(test_loss), debug='Loss Test {}:'.format(i))
            self.log['test_accuracy'].write('{}'.format(test_accuracy), debug='Accuracy Test {}:'.format(i))
