import torch
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

class DistributedDataManager:
    """Manages distributed data loading and synchronization."""
    def __init__(self, datasets, batch_size, num_workers=4):
        self.datasets = datasets
        self.batch_size = batch_size
        self.num_workers = num_workers

    def create_distributed_dataloaders(self):
        """Create data loaders for distributed training."""
        dataloaders = []

        for dataset in self.datasets:
            # Create a distributed sampler
            sampler = DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=True
            )

            # Create a data loader
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True
            )

            dataloaders.append(dataloader)

        return dataloaders

    def synchronize_datasets(self):
        """Ensure all nodes have consistent dataset ordering by synchronizing random seeds."""
        # Create a tensor to hold the seed on the current device
        seed_tensor = torch.zeros(1, dtype=torch.long, device=torch.cuda.current_device())

        if dist.get_rank() == 0:
            # Generate a seed only on the master process
            seed = torch.randint(0, 2**32, (1,)).item()
            seed_tensor[0] = seed

        # Broadcast the seed from the master process to all other processes
        dist.broadcast(seed_tensor, src=0)

        # Use the broadcasted seed
        seed_item = seed_tensor.item()
        torch.manual_seed(seed_item)
        np.random.seed(seed_item)
