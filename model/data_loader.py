import random
import os
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


class CarvanaDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """

    def __init__(self, filenames, transform, imgdir, maskdir=None):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        def img2mask_name(img_name):
            return img_name[:-4] + '_mask.gif'

        self.imgnames = filenames
        self.imgdir = imgdir

        self.masknames = list(map(img2mask_name, filenames)) if maskdir is not None else None
        self.maskdir = maskdir

        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.imgnames)


    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        img_i = self.imgnames[idx]
        img_path = os.path.join(self.imgdir, img_i)
        image = Image.open(img_path)  # PIL image

        if self.masknames is not None:
            mask_i = self.masknames[idx]
            mask_path = os.path.join(self.maskdir, mask_i)
            mask = Image.open(mask_path).convert('1')

            # apply same random argumentation to mask and image
            seed = random.randint(0, 2**32)
            random.seed(seed)
            image = self.transform(image)
            random.seed(seed)
            mask = self.transform(mask)

            return image, mask

        return self.transform(image)

# dir_path = os.path.dirname(os.path.realpath(__file__))
# parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))

# data_dir = os.path.join(parent_path, 'data')


def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    train_transformer = transforms.Compose([
        # resize the image to 64x64 (remove if images are already 64x64)
        # transforms.Resize(256),
        # transforms.RandomRotation(0.0),
        transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
        transforms.Resize((params.image_size, params.image_size)),
        transforms.ToTensor()])  # transform it into a torch tensor

    # loader for evaluation, no horizontal flip
    eval_transformer = transforms.Compose([
        # resize the image to 64x64 (remove if images are already 64x64)
        transforms.Resize((params.image_size, params.image_size)),
        transforms.ToTensor()])  # transform it into a torch tensor

    train_dir = os.path.join(data_dir, 'train')
    train_masks_dir = os.path.join(data_dir, 'train_masks')
    test_dir = os.path.join(data_dir, 'test')

    test_names = os.listdir(test_dir)

    # metadata_csv = os.path.join(data_dir, 'metadata.csv')
    train_masks_csv = os.path.join(data_dir, 'train_masks.csv')
    # metadata_df = pd.read_csv(metadata_csv)
    train_masks_df = pd.read_csv(train_masks_csv)
    imgs_name = train_masks_df['img'].tolist()
    train_names, val_names = train_test_split(imgs_name, test_size=0.2, shuffle=True, random_state=42)

    dataloaders = {}
    assert set(types) <= set(['train', 'val', 'test']), "data types have to be among {'train', 'val', 'test'}"
    for split in set(types):
        # use the train_transformer if training data, else use eval_transformer without random flip
        if split == 'train':
            dl = DataLoader(CarvanaDataset(train_names, train_transformer, train_dir, train_masks_dir),
                            batch_size=params.batch_size,
                            shuffle=True,
                            num_workers=params.num_workers,
                            pin_memory=params.cuda)
            dataloaders[split] = dl
        elif split == 'val':
            dl = DataLoader(CarvanaDataset(val_names, eval_transformer, train_dir, train_masks_dir),
                            batch_size=params.batch_size,
                            shuffle=False,
                            num_workers=params.num_workers,
                            pin_memory=params.cuda)
            dataloaders[split] = dl
        else:
            dl = DataLoader(CarvanaDataset(test_names, eval_transformer, test_dir),
                            batch_size=params.batch_size,
                            shuffle=False,
                            num_workers=params.num_workers,
                            pin_memory=params.cuda)
            dataloaders[split] = dl

    return dataloaders
