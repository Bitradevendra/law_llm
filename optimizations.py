import torch
import torch.distributed as dist

class GradientCompression:
    """Compresses gradients using top-k selection to reduce communication overhead."""
    def __init__(self, compression_ratio=0.001):
        self.compression_ratio = compression_ratio

    def compress_gradients(self, model):
        """Compresses gradients of a model using top-k selection."""
        compressed_grads = {}
        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            grad = param.grad.data
            flat_grad = grad.flatten()

            k = max(1, int(len(flat_grad) * self.compression_ratio))
            _, indices = torch.topk(torch.abs(flat_grad), k)

            values = flat_grad[indices]
            compressed_grads[name] = {
                'indices': indices,
                'values': values,
                'shape': grad.shape
            }
        return compressed_grads

    def decompress_and_apply_gradients(self, model, compressed_grads):
        """Decompresses gradients and applies them to the model."""
        for name, param in model.named_parameters():
            param.grad = torch.zeros_like(param.data)
            if name in compressed_grads:
                compressed = compressed_grads[name]
                flat_grad = torch.zeros(torch.prod(torch.tensor(compressed['shape'])).item(), device=compressed['values'].device)
                flat_grad[compressed['indices']] = compressed['values']
                param.grad.data = flat_grad.reshape(compressed['shape'])

class AdaptiveBatchSizing:
    """Dynamically adjusts the batch size based on available GPU memory."""
    def __init__(self, initial_batch_size=32, max_memory_usage=0.9, min_batch_size=2, max_batch_size=128):
        self.current_batch_size = initial_batch_size
        self.max_memory_usage = max_memory_usage
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_history = []

    def adjust_batch_size(self):
        """Adjusts batch size based on recent memory usage."""
        current_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        self.memory_history.append(current_memory)

        if len(self.memory_history) > 10:
            self.memory_history.pop(0)

        avg_memory = sum(self.memory_history) / len(self.memory_history)

        if avg_memory < 0.7 and self.current_batch_size < self.max_batch_size:
            self.current_batch_size = min(self.max_batch_size, int(self.current_batch_size * 1.2))
        elif avg_memory > self.max_memory_usage:
            self.current_batch_size = max(self.min_batch_size, int(self.current_batch_size * 0.8))

        return self.current_batch_size

class OverlappedTraining:
    """Overlaps gradient computation with communication to improve training efficiency."""
    def __init__(self, model):
        self.model = model
        self.communication_stream = torch.cuda.Stream()

    def overlapped_backward_pass(self, loss):
        """Performs a backward pass with overlapped communication."""
        loss.backward()

        with torch.cuda.stream(self.communication_stream):
            self.async_gradient_sync()

        torch.cuda.current_stream().wait_stream(self.communication_stream)

    def async_gradient_sync(self):
        """Asynchronously synchronizes gradients across all processes."""
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, async_op=True)
