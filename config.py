class ThreeDParallelismConfig:
    """Configuration for 3D parallelism."""
    def __init__(self):
        self.data_parallel_size = 2      # 2 nodes
        self.tensor_parallel_size = 1    # Within node parallelism
        self.pipeline_parallel_size = 2  # Pipeline stages

    def calculate_total_gpus(self):
        return (self.data_parallel_size *
                self.tensor_parallel_size *
                self.pipeline_parallel_size)

def setup_deepspeed_config():
    """Configure DeepSpeed for billion-parameter training."""
    config = {
        "train_batch_size": 32,
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 8,

        "zero_optimization": {
            "stage": 3,  # ZeRO-3: partition parameters, gradients, optimizer
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e5
        },

        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },

        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 1e-4,
                "warmup_num_steps": 1000
            }
        }
    }
    return config
