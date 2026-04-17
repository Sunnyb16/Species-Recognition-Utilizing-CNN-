#model creation
#optimizer
#loss
#training loop
#validation loop
from torch.utils.data import DataLoader
import src.precompute as pc
from src.dataset import BirdChunkDataset

def get_fold_loader(
    fold,
    file_level_df,
    master_df,
    templates,
    encode_fn,
    batch_size=32,
):
    train_samples, val_samples = pc.precompute_fold(
        fold=fold,
        file_level_df=file_level_df,
        master_df=master_df,
        templates=templates,
        encode_labels=encode_fn
    )

    train_ds = BirdChunkDataset(train_samples, augment=True)
    val_ds   = BirdChunkDataset(val_samples, augment=False)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    )