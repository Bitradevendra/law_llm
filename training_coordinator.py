import torch
import torch.nn.functional as F
import asyncio

class FederatedTrainingCoordinator:
    """Coordinates federated learning across multiple nodes with different datasets."""
    def __init__(self, nodes, datasets, global_model):
        self.nodes = nodes
        self.datasets = datasets
        self.global_model = global_model

    def federated_averaging(self, local_models, weights):
        """Aggregates model parameters using federated averaging."""
        global_dict = self.global_model.state_dict()

        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
            for model, weight in zip(local_models, weights):
                global_dict[key] += weight * model.state_dict()[key]

        return global_dict

    async def train_local_model(self, node_id, dataset, model):
        """Placeholder for training a model on a local node."""
        # This function should contain the actual training loop for a single node.
        # It would typically be a more complex function involving a training loop.
        print(f"Training on node {node_id} with {len(dataset)} samples.")
        # Returning the model for demonstration purposes.
        return model

    async def coordinate_training_round(self):
        """Coordinates one round of federated training."""
        local_models = []

        # Train on each node with a different dataset
        tasks = []
        for node_id, dataset in enumerate(self.datasets):
            task = self.train_local_model(
                node_id, dataset, self.global_model.cpu().clone()  # Send a copy
            )
            tasks.append(task)
        
        local_models = await asyncio.gather(*tasks)

        # Aggregate models
        dataset_sizes = [len(dataset) for dataset in self.datasets]
        total_size = sum(dataset_sizes)
        weights = [size / total_size for size in dataset_sizes]

        aggregated_params = self.federated_averaging(local_models, weights)

        # Update global model
        self.global_model.load_state_dict(aggregated_params)

        return self.global_model


class CrossDatasetDistillation:
    """Merges knowledge from models trained on different datasets using distillation."""
    def __init__(self, teacher_models, student_model, temperature=4.0, alpha=0.7):
        self.teachers = teacher_models
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha

    def ensemble_teacher_predictions(self, inputs):
        """Combines predictions from multiple teacher models."""
        teacher_outputs = []
        for teacher in self.teachers:
            teacher.eval()
            with torch.no_grad():
                output = teacher(inputs)
                teacher_outputs.append(output)

        # Average teacher predictions
        ensemble_output = torch.stack(teacher_outputs).mean(dim=0)
        return ensemble_output

    def distillation_loss(self, student_logits, teacher_logits, true_labels):
        """Computes the combined distillation and task loss."""
        # Soft target loss (knowledge distillation)
        student_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)

        kd_loss = F.kl_div(
            student_probs, teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Hard target loss (original task)
        ce_loss = F.cross_entropy(student_logits, true_labels)

        # Combined loss
        return self.alpha * kd_loss + (1 - self.alpha) * ce_loss
