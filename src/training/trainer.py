"""
Trainer Pipeline for NarrativeGraph.
Manages training, validation, optimization, learning rate schedules, mixed precision, and checkpointing.
"""

import os
from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from src.training.loss import NarrativeGraphMultiTaskLoss
from src.evaluation.metrics import compute_all_metrics


class NarrativeGraphTrainer:
    """
    Standardized Trainer for NarrativeGraph experiments.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        epochs: int = 5,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 4,
        fp16: bool = True,
        save_dir: str = "checkpoints",
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.fp16 = fp16
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.loss_fn = NarrativeGraphMultiTaskLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        total_steps = (len(train_dataloader) // gradient_accumulation_steps + 1) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=max(1, total_steps)
        )
        self.scaler = torch.amp.GradScaler('cuda', enabled=fp16 and torch.cuda.is_available())

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_dataloader):
            batch_dev = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with torch.amp.autocast('cuda', enabled=self.fp16 and torch.cuda.is_available()):
                outputs = self.model(batch_dev)
                loss, loss_dict = self.loss_fn(outputs, batch_dev)
                loss = loss / self.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(self.train_dataloader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

            total_loss += loss.item() * self.gradient_accumulation_steps

        return {"train_loss": total_loss / len(self.train_dataloader)}

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_outputs = []
        all_targets = []
        total_val_loss = 0.0

        import gc
        with torch.no_grad():
            for batch in dataloader:
                batch_dev = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                outputs = self.model(batch_dev)
                loss, _ = self.loss_fn(outputs, batch_dev)
                total_val_loss += loss.item()

                # Detach to CPU to prevent RAM / VRAM memory growth
                cpu_outputs = {
                    k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in outputs.items()
                }
                cpu_targets = {
                    k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in batch_dev.items()
                }
                all_outputs.append(cpu_outputs)
                all_targets.append(cpu_targets)

        metrics = compute_all_metrics(all_outputs, all_targets)
        metrics["val_loss"] = total_val_loss / len(dataloader)
        
        # Immediate memory cleanup
        del all_outputs, all_targets
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return metrics

    def fit(self) -> Dict[str, float]:
        import gc
        best_val_f1 = 0.0
        best_metrics = {}

        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_epoch()
            val_metrics = self.evaluate(self.val_dataloader)
            print(f"[Epoch {epoch}/{self.epochs}] Train Loss: {train_metrics['train_loss']:.4f} | Val F1: {val_metrics.get('macro_f1', 0.0):.4f}")

            val_f1 = val_metrics.get("macro_f1", 0.0)
            if val_f1 >= best_val_f1:
                best_val_f1 = val_f1
                best_metrics = val_metrics
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_model.pt"))

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return best_metrics
