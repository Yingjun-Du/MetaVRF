import sys
import omniglot
import mini_imagenet
import tiered_imagenet

import cifar_fs

"""
General function that selects and initializes the particular dataset to use for
few-shot classification. Additional dataset support should be added here.
"""


def get_data(dataset, mode='train', seed=1):
    if dataset == 'Omniglot':
        return omniglot.OmniglotData(path='../data/omniglot.npy',
                                     train_size=1100,
                                     validation_size=100,
                                     augment_data=True,
                                     seed=seed)
    elif dataset == 'miniImageNet':
        return mini_imagenet.MiniImageNetData(path='../data', seed=seed)
    elif dataset == 'cifarfs':
        return cifar_fs.CifarData(path='../data', seed=seed)
    elif dataset == 'tieredImageNet':
        return tiered_imagenet.tieredImageNetData(path='../data', seed=seed)

    else:
        sys.exit("Unsupported dataset type (%s)." % dataset)
