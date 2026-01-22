from torch.utils.data import DataLoader


def create_dataloaders(
    train_ds, val_ds, test_ds, batch_size=64, num_workers=2, shuffle_val=True
):
    return (
        DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        DataLoader(
            val_ds, batch_size=batch_size, shuffle=shuffle_val, num_workers=num_workers
        ),
        DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
    )
