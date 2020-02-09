# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from . import additional_transforms as add_transforms
from abc import abstractmethod

import os
import glob
from torchvision.datasets.utils import download_url, check_integrity
import torch.utils.data as data

class Caltech256(data.Dataset):
  """`Caltech256.
  Args:
      root (string): Root directory of dataset where directory
          ``256_ObjectCategories`` exists.
      train (bool, optional): Not used
      transform (callable, optional): A function/transform that  takes in an PIL image
          and returns a transformed version. E.g, ``transforms.RandomCrop``
      target_transform (callable, optional): A function/transform that takes in the
          target and transforms it.
      download (bool, optional): If true, downloads the dataset from the internet and
          puts it in root directory. If dataset is already downloaded, it is not
          downloaded again.
  """
  base_folder = '256_ObjectCategories'
  url = "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar"
  filename = "256_ObjectCategories.tar"
  tgz_md5 = '67b4f42ca05d46448c6bb8ecd2220f6d'

  def __init__(self, root, train=True,
               transform=None, target_transform=None,
               download=False):
    self.root = os.path.expanduser(root)
    self.transform = transform
    self.target_transform = target_transform

    if download:
      self.download()

    if not self._check_integrity():
      raise RuntimeError('Dataset not found or corrupted.' +
                         ' You can use download=True to download it')

    self.data = []
    self.labels = []

    for cat in range(0, 257):
      print (cat)

      cat_dirs = glob.glob(os.path.join(self.root, self.base_folder, '%03d*' % cat))

      for fdir in cat_dirs:
        for fimg in glob.glob(os.path.join(fdir, '*.jpg')):
            img = Image.open(fimg).convert("RGB")

            self.data.append(img)
            self.labels.append(cat)

  def __getitem__(self, index):
    """
    Args:
        index (int): Index
    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target = self.data[index], self.labels[index]

    #img = Image.fromarray(img)

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

  def __len__(self):
    return len(self.data)

  def _check_integrity(self):
    fpath = os.path.join(self.root, self.filename)
    if not check_integrity(fpath, self.tgz_md5):
      return False
    return True

  def download(self):
    import tarfile

    root = self.root
    download_url(self.url, root, self.filename, self.tgz_md5)

    # extract file
    cwd = os.getcwd()
    tar = tarfile.open(os.path.join(root, self.filename), "r")
    os.chdir(root)
    tar.extractall()
    tar.close()
    os.chdir(cwd)

  def __repr__(self):
    fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
    fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
    fmt_str += '    Root Location: {}\n'.format(self.root)
    tmp = '    Transforms (if any): '
    fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    tmp = '    Target Transforms (if any): '
    fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    return fmt_str


identity = lambda x:x
class SimpleDataset:
    def __init__(self, transform, target_transform=identity):
        self.transform = transform
        self.target_transform = target_transform


        self.d = Caltech256(root='./', transform=transform, target_transform=target_transform, download=True)
        #for i, (data, label) in enumerate(d):
        #    self.meta['image_names'].append(data)
        #    self.meta['image_labels'].append(label)      

    def __getitem__(self, i):

        #img = self.transform(self.meta['image_names'][i])
        #target = self.target_transform(self.meta['image_labels'][i])
        img, target = self.d[i]

        return img, target

    def __len__(self):
        #return len(self.meta['image_names'])
        return len(self.d)


class SetDataset:
    def __init__(self, batch_size, transform):

        self.sub_meta = {}
        self.cl_list = range(257)

        for cl in self.cl_list:
            self.sub_meta[cl] = []

        d = Caltech256(root='./', download=False)
        for i, (data, label) in enumerate(d):
            self.sub_meta[label].append(data)
    
        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform )
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self,i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.sub_dataloader)

class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,i):

        img = self.transform(self.sub_meta[i])
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = add_transforms.ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomSizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Scale':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Scale','CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager(object):
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 

class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size):        
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(transform)

        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 12, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

class SetDataManager(DataManager):
    def __init__(self, image_size, n_way=5, n_support=5, n_query=16, n_eposide = 100):        
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset(self.batch_size, transform)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )  
        data_loader_params = dict(batch_sampler = sampler,  num_workers = 12, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

if __name__ == '__main__':
    pass