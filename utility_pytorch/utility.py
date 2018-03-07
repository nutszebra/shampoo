import os
import six
from tqdm import tqdm
import six.moves.cPickle as pickle


def make_dir_one(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_dir(path):
    separated_path = path.split('/')
    tmp_path = ''
    for directory in separated_path:
        tmp_path = tmp_path + directory + '/'
        if directory == '.':
            continue
        make_dir_one(tmp_path)
    return True


def find_files(path, affix_flag=False):
    if path[-1] == '/':
        path = path[:-1]
    if affix_flag is False:
        return [path + '/' + name for name in os.listdir(path)]
    else:
        return [name for name in os.listdir(path)]


def remove_slash(path):
    return path[:-1] if path[-1] == '/' else path


def load_pickle(path, encoding=None):
    with open(path, 'rb') as f:
        if encoding is None:
            answer = pickle.load(f)
        else:
            answer = pickle.load(f, encoding=encoding)
    return answer


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return True


def create_progressbar(end, desc='', stride=1, start=0, dynamic_ncols=True, ncols=80):
    if type(end) is int or type(end) is float:
        return tqdm(six.moves.range(int(start), int(end), int(stride)), desc=desc, leave=False, ncols=ncols)
    else:
        return tqdm(end, desc=desc, leave=False, ncols=ncols)


def write(*args, **kwargs):
    tqdm.write(*args, **kwargs)
