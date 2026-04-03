import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def setup_distributed(rank, world_size, master_ip="localhost"):
    """Initialize distributed training environment."""
    # Set up process group for multi-node communication
    init_process_group(
        backend='nccl',  # NVIDIA GPU backend
        init_method=f'tcp://{master_ip}:12355',
        rank=rank,
        world_size=world_size
    )
    # Set device for current process
    torch.cuda.set_device(rank % torch.cuda.device_count())

def create_distributed_model(model, device_id):
    """Wrap model with DDP for distributed training."""
    model = model.to(device_id)
    ddp_model = DDP(
        model,
        device_ids=[device_id],
        find_unused_parameters=False,  # Optimization for performance
        gradient_as_bucket_view=True   # Memory optimization
    )
    return ddp_model
