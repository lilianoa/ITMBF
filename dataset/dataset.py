"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import itertools
import json
import os
import pickle
from typing import Dict, List, Tuple
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tool import utils

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('.', ' ').replace(';', ' ').replace('(', ' ').replace(')', ' ').replace('-', ' ').replace('?', '').replace('\'s', ' \'s').replace('s\'', ' ').replace('n\'t', ' not').replace('"', ' ')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))   # 在add_word里实现了word2idx、idx2word
        else:
            for w in words:
                # the least frequent word (`bebe`) as UNK for Visual Genome datasets
                tokens.append(self.word2idx.get(w, self.padding_idx-1))
        return tokens

    def dump_to_file(self, path):
        pickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = pickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

def tfidf_from_questions(names, dictionary, dataroot='data'):
    inds = [[], []]   # rows, cols for uncoalesce sparse matrix
    df = dict()
    N = len(dictionary)

    def populate(inds, df, text):
        tokens = dictionary.tokenize(text, True)
        for t in tokens:
            df[t] = df.get(t, 0) + 1
        combin = list(itertools.combinations(tokens, 2))
        for c in combin:
            if c[0] < N:
                inds[0].append(c[0]); inds[1].append(c[1])
            if c[1] < N:
                inds[0].append(c[1]); inds[1].append(c[0])
    for split in names:
        captions = json.load(open(os.path.join(dataroot, 'annotations', "%s_captions.json" % split), 'r'))
        for caps in captions['captions']:
            populate(inds, df, caps['caption'])

    # TF-IDF
    vals = [1] * len(inds[1])
    for idx, col in enumerate(inds[1]):
        assert df[col] >= 1, 'document frequency should be greater than zero!'
        vals[col] /= df[col]

    # Make stochastic matrix
    def normalize(inds, vals):
        z = dict()
        for row, val in zip(inds[0], vals):
            z[row] = z.get(row, 0) + val
        for idx, row in enumerate(inds[0]):
            vals[idx] /= z[row]
        return vals

    vals = normalize(inds, vals)

    tfidf = torch.sparse.FloatTensor(torch.LongTensor(inds), torch.FloatTensor(vals))
    tfidf = tfidf.coalesce()

    # Latent word embeddings
    emb_dim = 300
    glove_file = 'data/glove/glove.6B.%dd.txt' % emb_dim
    weights, word2emb = utils.create_glove_embedding_init(dictionary.idx2word[N:], glove_file)
    print('tf-idf stochastic matrix (%d x %d) is generated.' % (tfidf.size(0), tfidf.size(1)))

    return tfidf, weights



# 创建读取带说明的胃镜图像的Dataset类
class GasCapDataset(Dataset):
    def __init__(self, split, dictionary, dataroot="data", transform=None, nans=2, max_length=48):
        super(GasCapDataset).__init__()
        assert split in ['train', 'val', 'test']
        self.dictionary = dictionary
        self.root_dir = dataroot
        self.split = split
        self.transform = transform
        self.class_to_idx = {'y': 1, 'n': 0}
        self.num_ans_candidates = nans
        self.max_length = max_length

        self.entries = []
        entry = {}
        caption_path = os.path.join(dataroot, 'annotations', "%s_captions.json" % self.split)
        captions = json.load(open(caption_path))['captions']
        for cap in captions:
            img_id = cap['image_id']
            caption = cap['caption']
            label_str = cap['label']
            label_id = self.class_to_idx[label_str]
            img_path = os.path.join(self.root_dir, self.split, label_str, img_id+".jpeg")
            entry['img_path'] = img_path
            entry['label_id'] = label_id
            entry['caption'] = caption
            self.entries.append(entry.copy())

        self.tokenize()
        self.tensorize()

    def tokenize(self):
        """Tokenizes the captions.

        This will add cap_token in each entry of the datasets.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        max_length = self.max_length
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['caption'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['cap_token'] = tokens

    def tensorize(self):
        # 没有把img tensor化，因此在读取img之后需要toTensor，在self.transform中需要toTensor
        for entry in self.entries:
            caption = torch.from_numpy(np.array(entry['cap_token']))
            entry['cap_token'] = caption
            label_id = entry['label_id']
            if None!=label_id:
                label_id = torch.from_numpy(np.array(entry['label_id'])).long()
                entry['label_id'] = label_id

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a datasets.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        item = {}
        entry = self.entries[index]
        img_path = entry["img_path"]
        img = Image.open(img_path)
        if self.transform is not None:
            item['img'] = self.transform(img)
        item['caption'] = entry["cap_token"]
        item['label_id'] = entry["label_id"]
        return item

    def __len__(self):
        return len(self.entries)


if __name__ == '__main__':
    dict_path = "./gas_data/gas_data/dictionary.pkl"
    dataroot = "./gas_data/gas_data"
    dictionary = Dictionary.load_from_file(dict_path)
    transform = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor()
    ])
    train_dataset = GasCapDataset(split="train", dictionary=dictionary, dataroot=dataroot, transform=transform)
    loader = DataLoader(train_dataset, 10, shuffle=True, num_workers=4)
    for i, batch in enumerate(loader):
        print(batch['img'].size())
        print(batch['caption'].shape)
        print(batch['label_id'])
        print('----------------')