import torch
import torch.distributed as dist
import numpy as np

try:
    import wandb
except ImportError:
    wandb = None

class TrainingMonitor:
    """Monitors training efficiency metrics like throughput and communication overhead."""
    def __init__(self):
        self.metrics = {
            'throughput': [],
            'memory_usage': [],
            'communication_time': [],
            'compute_time': []
        }

    def log_training_metrics(self, batch_time, communication_time,
                           memory_usage, throughput):
        """Log and analyze training performance."""
        self.metrics['throughput'].append(throughput)
        self.metrics['memory_usage'].append(memory_usage)
        self.metrics['communication_time'].append(communication_time)
        self.metrics['compute_time'].append(batch_time - communication_time)

        # Analyze performance every 100 steps
        if len(self.metrics['throughput']) % 100 == 0:
            self.analyze_performance()

    def analyze_performance(self):
        """Analyze performance bottlenecks and print warnings if necessary."""
        if dist.get_rank() == 0:  # Only print from the master process
            avg_comm_time = np.mean(self.metrics['communication_time'][-100:])
            avg_compute_time = np.mean(self.metrics['compute_time'][-100:])
            total_time = avg_comm_time + avg_compute_time

            if total_time > 0:
                comm_ratio = avg_comm_time / total_time
                if comm_ratio > 0.3:
                    print(f"Warning: Communication overhead is high: {comm_ratio:.2%}")
                    print("Consider using gradient compression or optimizing network topology.")

            avg_mem_usage = np.mean(self.metrics['memory_usage'][-100:])
            if avg_mem_usage < 0.7:
                print(f"Warning: Memory efficiency is low: {avg_mem_usage:.2%}")
                print("Consider increasing the batch size for better GPU utilization.")

class ProductionMonitoring:
    """Handles logging of distributed metrics to monitoring services like Weights & Biases."""
    def __init__(self, use_wandb=True):
        self.wandb_enabled = use_wandb and (wandb is not None)
        if self.wandb_enabled and dist.get_rank() == 0:
            print("Weights & Biases is enabled. Ensure you have initialized it in your main training script.")

    def log_distributed_metrics(self, metrics, step):
        """Log metrics from distributed training to a monitoring service."""
        if dist.get_rank() == 0:  # Only log from the master process
            aggregated_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    # Ensure the tensor is on the correct device for reduction
                    value_tensor = value.clone().detach().cuda()
                    dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
                    aggregated_metrics[key] = value_tensor.item() / dist.get_world_size()
                else:
                    # For non-tensor metrics, assume they are already aggregated or rank-0 specific
                    aggregated_metrics[key] = value

            if self.wandb_enabled:
                wandb.log(aggregated_metrics, step=step)
