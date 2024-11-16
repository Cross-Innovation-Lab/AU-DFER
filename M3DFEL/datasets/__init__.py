from .dataset_DFEW_AU_weighted import DFEWDataset
from .dataset_FERV39K_AU_weighted import FERV39KDataset
import torch


def create_dataloader(args, mode):
    """create dataloader according to args and training/testing mode

    Args:
        args
        mode: String("train" or "test")

    Returns:
        dataloader
    """
    if args.dataset=='DFEW':
        dataset = DFEWDataset(args, mode)
    elif args.dataset=='FERV39K':
        dataset = FERV39KDataset(args, mode)

    dataloader = None

    # return train_dataset or test_dataset according to the mode
    if mode == "train":
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 num_workers=args.workers,
                                                 pin_memory=True,
                                                 drop_last=True)
    elif mode == "test":
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 num_workers=args.workers,
                                                 pin_memory=True)
    return dataloader
