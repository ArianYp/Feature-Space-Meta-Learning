import os
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import torchvision.transforms as transforms
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import torchvision.transforms as transforms


class Dominoes (Dataset):
    def __init__(self, split, path, mask_path = None, get_mask = False, get_names = False, transform=None):
        self.get_mask = get_mask
        self.get_names = get_names
        self.split = split

        self.X = torch.tensor(np.load(os.path.join(path,f'X_{split}.npy')))
        self.y = F.one_hot(torch.tensor(np.load(os.path.join(path, f'y_{split}.npy'))).type(torch.LongTensor), 2).type(torch.FloatTensor)
        self.envs = F.one_hot(torch.tensor(np.load(os.path.join(path, f'env_{split}.npy'))).type(torch.LongTensor), 4)

        self.get_mask = get_mask
        self.get_names = get_names
        self.mask_path = mask_path
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if not self.transform==None:
            x = self.transform(self.X[idx])
        else:
            x = self.X[idx]
        ret = [x, self.y[idx], self.envs[idx]]

        if self.get_mask:
            mask = np.load(os.path.join(self.mask_path, f'{idx}.npy'))
            ret.append(mask)

        if self.get_names:
            ret.append(str(idx))

        return tuple(ret)


def get_domino_loaders(path, batch_size = 32, mask_path = None, get_mask = False, get_names = False, use_aug = False):
    trainset = Dominoes('train', path = path, mask_path=mask_path, get_mask = get_mask, get_names=get_names)
    valset = Dominoes('val', path=path, mask_path=mask_path, get_mask=get_mask, get_names=get_names)
    testset = Dominoes('test', path=path, mask_path=mask_path, get_mask=get_mask, get_names=get_names)


    trainloader = DataLoader(trainset, shuffle=True, num_workers=4,
                           batch_size=batch_size)
    valloader = DataLoader(valset, shuffle=False, num_workers=4,
                         drop_last=False, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=False, num_workers=4,
                          drop_last=False, batch_size=batch_size)

    return trainloader, valloader, testloader
from PIL import Image
from torch.utils.data import Dataset, DataLoader,Subset
from torch.utils.data.distributed import DistributedSampler
import random
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from torchvision.transforms.functional import to_pil_image

def identify_hard_samples(
    model,
    dataloader,
    device,
    n_samples_per_class=150,
    log_path: Path | None = None,
):
    """
    Identify samples with highest loss per class
    Returns indices of hard samples (dataset indices when available)
    """
    model.eval()
    losses_per_sample = []
    labels_per_sample = []
    indices_per_sample = []
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        # Prefer true dataset indices from the DataLoader's batch_sampler (robust to shuffle=True).
        # Fallback to a running counter if batch indices are not available.
        have_batch_indices = hasattr(dataloader, "batch_sampler") and dataloader.batch_sampler is not None
        idx_counter = 0

        it = zip(dataloader.batch_sampler, dataloader) if have_batch_indices else ((None, b) for b in dataloader)
        for batch_indices, (img1, img2, label, env) in it:
            img1 = img1.to(device)
            label_idx = label.argmax(dim=1).to(device)
            
            outputs = model(img1)
            loss = criterion(outputs, label_idx)
            
            batch_size = img1.size(0)
            for i in range(batch_size):
                losses_per_sample.append(loss[i].item())
                labels_per_sample.append(label_idx[i].item())
                if batch_indices is not None and i < len(batch_indices):
                    indices_per_sample.append(int(batch_indices[i]))
                else:
                    indices_per_sample.append(int(idx_counter))
                idx_counter += 1
    
    # Convert to arrays
    losses = np.array(losses_per_sample)
    labels = np.array(labels_per_sample)
    indices = np.array(indices_per_sample)
    
    # Get top-k hard samples per class
    hard_indices = []
    hard_rows = []
    for class_id in [0, 1]:
        class_mask = labels == class_id
        class_indices = indices[class_mask]
        class_losses = losses[class_mask]
        
        # Get indices of samples with highest loss
        k = int(min(int(n_samples_per_class), len(class_losses)))
        top_k_local = np.argsort(class_losses)[-k:]
        top_k_global = class_indices[top_k_local]
        hard_indices.extend(top_k_global.tolist())

        # Store for logging (sorted descending by loss)
        order = np.argsort(class_losses)[::-1]
        for rank, j in enumerate(order[:k]):
            hard_rows.append(
                {
                    "class_id": int(class_id),
                    "rank_in_class": int(rank),
                    "dataset_idx": int(class_indices[j]),
                    "loss": float(class_losses[j]),
                }
            )

    # Optional logging
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            pd.DataFrame(hard_rows).sort_values(["class_id", "rank_in_class"]).to_csv(log_path, index=False)
        except Exception:
            pass
        # Also print a compact summary for quick sanity checks
        try:
            df = pd.DataFrame(hard_rows)
            for cid in [0, 1]:
                dfc = df[df["class_id"] == cid].sort_values("loss", ascending=False)
                if len(dfc) == 0:
                    continue
                print(f"[hard samples] class {cid}: top-{len(dfc)} losses:")
                for _, r in dfc.iterrows():
                    print(f"  idx={int(r['dataset_idx'])} loss={float(r['loss']):.6f}")
        except Exception:
            pass
    
    return hard_indices

# Function to create subset of hard samples
def create_hard_sample_loader(dataset, hard_indices, batch_size):
    hard_dataset = Subset(dataset, hard_indices)
    kwargs = {'pin_memory': True, 'num_workers': 4, 'drop_last': False}
    return DataLoader(hard_dataset, batch_size=batch_size, shuffle=True, **kwargs),hard_dataset


def save_hard_samples(hard_dataset, out_dir: Path, max_to_save: int | None = None) -> None:
    """
    Save images from the hard-sample subset to disk for inspection.

    Expects dataset items shaped like: (img1, img2, label, env)
    and saves the `img2` view (typically the diffusion/generation view).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subset_indices = getattr(hard_dataset, "indices", None)  # Subset(dataset, indices)

    n = len(hard_dataset)
    if max_to_save is not None:
        n = min(n, int(max_to_save))

    rows = []
    for i in range(n):
        _img1, img2, label, _env = hard_dataset[i]
        class_id = int(label.argmax().item()) if torch.is_tensor(label) else int(label)
        orig_idx = int(subset_indices[i]) if subset_indices is not None else int(i)

        if torch.is_tensor(img2):
            img2_vis = img2.detach().cpu().clamp(0, 1)
            pil = to_pil_image(img2_vis)
        else:
            pil = img2

        fname = f"class{class_id}_orig{orig_idx}_subset{i}.png"
        fpath = out_dir / fname
        pil.save(fpath)

        rows.append(
            {
                "subset_idx": i,
                "orig_idx": orig_idx,
                "class_id": class_id,
                "path": str(fpath),
            }
        )

    # Manifest for traceability
    try:
        pd.DataFrame(rows).to_csv(out_dir / "manifest.csv", index=False)
    except Exception:
        pass

# Plot warmup training curve to check for convergence
# First, let's run a SHORT warmup with logging to collect loss history

class WaterbirdDataset(Dataset):
    
    def __init__(self, split, transform1, transform2, root_dir="../DaC/waterbird_complete95_forest2water2"):
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        self.env_dict = {
            (0, 0): torch.Tensor(np.array([1,0,0,0])),#Landbird with land background
            (0, 1): torch.Tensor(np.array([0,1,0,0])),#Landbird with Water background
            (1, 0): torch.Tensor(np.array([0,0,1,0])),#Waterdbird with Land background
            (1, 1): torch.Tensor(np.array([0,0,0,1])) #Waterbird with water background
        }
        self.split = split
        self.dataset_dir = root_dir
        if not os.path.exists(self.dataset_dir):
            raise ValueError(
                f'{self.dataset_dir} does not exist yet. Please generate the dataset first.')
        self.metadata_df = pd.read_csv(os.path.join(self.dataset_dir, 'metadata.csv'))
        self.metadata_df = self.metadata_df[self.metadata_df['split']==self.split_dict[self.split]]
        y_array = torch.Tensor(np.array(self.metadata_df['y'].values)).type(torch.LongTensor)
        self.y_array = self.metadata_df['y'].values

        self.place_array = self.metadata_df['place'].values
        self.filename_array = self.metadata_df['img_filename'].values
        self.transform1 = transform1
        self.transform2 = transform2

        self.y_one_hot = nn.functional.one_hot(y_array, num_classes=2).type(torch.FloatTensor)
       
    def __len__(self):
        return len(self.filename_array)

    
    def __getitem__(self, idx):
        y = self.y_array[idx]
        place = self.place_array[idx]
        img_filename = os.path.join(
            self.dataset_dir,
            self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')
        img1 = self.transform1(img)
        img2 = self.transform2(img)

        label = self.y_one_hot[idx]

        return img1, img2, label, self.env_dict[(y, place)]

    def get_raw_image(self,idx):
      scale = 256.0/224.0
      target_resolution = [224, 224]
      img_filename = os.path.join(
            self.dataset_dir,
            self.filename_array[idx])
      img = Image.open(img_filename).convert('RGB')
      transform = transforms.Compose([
          transforms.Resize((768, 512)),
          transforms.ToTensor()
      ])
      return transform(img)

import torchvision

class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = torchvision.models.resnet50(pretrained=True)
        d = self.model.fc.in_features
        self.model.fc = nn.Linear(d, 2)
    
    def forward (self, X):
        X = self.model(X)
        return X

def get_waterbird_dataloader(split, transform1, transform2, batch_size):
    kwargs = {'pin_memory': True, 'num_workers': 4, 'drop_last': False}
    dataset = WaterbirdDataset( split=split, transform1= transform1, transform2=transform2)
    if not split == 'train':
      print (split)
      dataloader = DataLoader(dataset=dataset,batch_size=batch_size, shuffle=False, **kwargs)
    else:
      dataloader = DataLoader(dataset=dataset,batch_size=batch_size, shuffle=True, **kwargs)
    return dataloader

def get_waterbird_dataset(split, transform1, transform2, root_dir="../DaC/waterbird_complete95_forest2water2"):
    dataset = WaterbirdDataset(
        split=split, transform1=transform1, transform2=transform2, root_dir=root_dir
    )
    return dataset

target_resolution = [224, 224]
def get_transform_cub(train):
    scale = 256.0/224.0
    target_resolution = [224, 224]
    assert target_resolution is not None

    if (not train):
      transform = transforms.Compose([
                transforms.Resize(
                    (int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
                transforms.CenterCrop(target_resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                0.229, 0.224, 0.225]),
            ])

    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [
                            0.229, 0.224, 0.225]),
        ])

    return transform

if __name__ == "__main__":
    preprocess = transforms.Compose([
            transforms.Resize((768, 768)),
            transforms.ToTensor(),  # Normalize to [-1, 1]
        ])
    args = {'batch_size': 64}
    t_train =  get_transform_cub(True)

    t_tst = get_transform_cub(False)
    traindset = get_waterbird_dataset('train',t_train, preprocess)
    valdset = get_waterbird_dataset('val',t_tst, preprocess)

    trainloader = get_waterbird_dataloader('train', t_train, preprocess, args['batch_size'])
    valloader = get_waterbird_dataloader('val', t_tst, preprocess, args['batch_size'])
    testloader = get_waterbird_dataloader('test', t_tst,preprocess, args['batch_size'])
    retrainloader =  get_waterbird_dataloader('train', t_tst, preprocess, args['batch_size'])

