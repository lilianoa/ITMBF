import os
from torchvision import transforms
from bootstrap.lib.options import Options
from datasets.dataset import Dictionary, GasCapDataset
def factory(engine=None):
    opt = Options()['dataset']
    dataset = {}  # dataset is a dictionary that contains all the needed datasets indexed by modes, example: train, eval
    if opt.get('train_split', None):
        dataset['train'] = factory_split(opt['train_split'])
    if opt.get('eval_split', None):
        dataset['eval'] = factory_split(opt['eval_split'])
    return dataset

def factory_split(split):
    opt = Options()['dataset']
    dataroot = opt['dir']
    dict_path = os.path.join(dataroot, 'dictionary.pkl')
    dictionary = Dictionary.load_from_file(dict_path)
    transform = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
    dataset = GasCapDataset(split=split, dictionary=dictionary, dataroot=dataroot, transform=transform)
    return dataset