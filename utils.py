import os
import time
import torch
import torch.distributed as dist

class DistributedCheckpointing:
    """Manages robust checkpointing for long, distributed training runs."""
    def __init__(self, checkpoint_dir, save_interval=1000):
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval
        if dist.get_rank() == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_distributed_checkpoint(self, model, optimizer, epoch, step):
        """Saves a checkpoint with the distributed training state."""
        if dist.get_rank() == 0:  # Only rank 0 saves the checkpoint
            checkpoint = {
                'epoch': epoch,
                'step': step,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'random_state': torch.get_rng_state(),
                'cuda_random_state': torch.cuda.get_rng_state(),
            }

            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f'checkpoint_epoch_{epoch}_step_{step}.pt'
            )
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            self.cleanup_old_checkpoints()

    def load_distributed_checkpoint(self, model, optimizer, checkpoint_path):
        """Loads a checkpoint and restores the training state."""
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{dist.get_rank()}')

        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        torch.set_rng_state(checkpoint['random_state'])
        torch.cuda.set_rng_state(checkpoint['cuda_random_state'])

        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch'], checkpoint['step']

    def cleanup_old_checkpoints(self, keep_last=5):
        """Removes old checkpoints to save space."""
        checkpoints = sorted(
            [os.path.join(self.checkpoint_dir, f) for f in os.listdir(self.checkpoint_dir)],
            key=os.path.getmtime
        )
        if len(checkpoints) > keep_last:
            for old_checkpoint in checkpoints[:-keep_last]:
                os.remove(old_checkpoint)
                print(f"Removed old checkpoint: {old_checkpoint}")

class FaultTolerantTraining:
    """Implements fault tolerance and recovery for distributed training."""
    def __init__(self, checkpoint_manager, setup_distributed_fn):
        self.checkpoint_manager = checkpoint_manager
        self.setup_distributed = setup_distributed_fn # Function to re-initialize process group
        self.health_check_interval = 100

    def health_check(self):
        """Checks the health of all training nodes."""
        try:
            test_tensor = torch.ones(1, device='cuda')
            dist.all_reduce(test_tensor)
            return True
        except Exception as e:
            print(f"Health check failed on rank {dist.get_rank()}: {e}")
            return False

    def recover_from_failure(self, last_checkpoint_path, world_size):
        """Recovers training from a node failure."""
        print("Attempting recovery from node failure...")
        dist.destroy_process_group()
        time.sleep(5)  # Wait for cleanup

        try:
            # The user needs to handle the restart of processes and provide the new world_size
            self.setup_distributed(rank=dist.get_rank(), world_size=world_size)
            self.checkpoint_manager.load_distributed_checkpoint(last_checkpoint_path)
            print("Recovery successful.")
            return True
        except Exception as e:
            print(f"Recovery failed: {e}")
            return False

def optimize_communication_topology(world_size, gpus_per_node):
    """Optimizes inter-node communication by creating hierarchical process groups."""
    rank = dist.get_rank()
    node_id = rank // gpus_per_node

    # Intra-node group
    for i in range(world_size // gpus_per_node):
        intra_node_ranks = list(range(i * gpus_per_node, (i + 1) * gpus_per_node))
        group = dist.new_group(intra_node_ranks)
        if i == node_id:
            local_group = group

    # Inter-node group
    inter_node_ranks = [i * gpus_per_node for i in range(world_size // gpus_per_node)]
    inter_node_group = dist.new_group(inter_node_ranks)

    return local_group, inter_node_group

def hierarchical_allreduce(tensor, local_group, inter_node_group):
    """Performs a hierarchical all-reduce for better bandwidth utilization."""
    rank = dist.get_rank()
    local_rank = rank % dist.get_group_rank(local_group, 0) # This is a simplification

    # Step 1: Reduce within the node
    dist.all_reduce(tensor, group=local_group)

    # Step 2: Inter-node communication (only for the first rank of each node)
    if local_rank == 0:
        dist.all_reduce(tensor, group=inter_node_group)

    # Step 3: Broadcast result within the node
    dist.broadcast(tensor, src=local_rank, group=local_group)
