import numpy as np
import os
import random
import time
import torch
import torchvision
import torchvision.transforms as transforms


def get_transforms(dataset, T=1.):
    transform_train = None
    transform_test = None

    if dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (2.023, 1.994, 2.010)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023*T, 0.1994*T, 0.2010*T)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (2.023, 1.994, 2.010)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023*T, 0.1994*T, 0.2010*T)),
        ])

    if dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675*T, 0.2565*T, 0.2761*T)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675*T, 0.2565*T, 0.2761*T)),
        ])

    if dataset == 'fashion_mnist':
        transform_train = transforms.Compose([
            transforms.Resize(size=32),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5,), (0.5*T,))
            ])
        transform_test = transforms.Compose([
            transforms.Resize(size=32),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5,), (0.5*T,))
            ])

    assert transform_test is not None and transform_train is not None, 'Error, no dataset %s' % dataset
    return transform_train, transform_test


def get_dataloader(dataset, train_batch_size, test_batch_size, scale=1., num_workers=2, root='../data'):
    transform_train, transform_test = get_transforms(dataset, T=scale)
    trainset, testset = None, None
    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

    if dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)

    if dataset == 'fashion_mnist':
        trainset = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform_test)

    assert trainset is not None and testset is not None, 'Error, no dataset %s' % dataset
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True,
                                              num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False,
                                             num_workers=num_workers)

    return trainloader, testloader


def set_random_seed(seed):
    """Set a seed for deterministic behaviors.
    Note: If someone runs an experiment with a pre-selected manual seed, he can
    definitely reproduce the results with the same seed; however, if he runs the
    experiment with seed=None and re-run the experiments using the seed previously
    returned from this function (e.g. the returned seed might be logged to
    Tensorboard), and if cudnn is used in the code, then there is no guarantee
    that the results will be reproduced with the recovered seed.
    Args:
        seed (int|None): seed to be used. If None, a default seed based on
            pid and time will be used.
    Returns:
        The seed being used if ``seed`` is None.
    """
    if seed is None:
        seed = int(np.uint32(hash(str(os.getpid()) + '|' + str(time.time()))))
    else:
        # torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
