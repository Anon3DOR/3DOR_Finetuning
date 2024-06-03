import torch
from torch.nn import functional as F

from source.models import model_base


class ClassificationModel(model_base.ModelBase):

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            xyz: Tensor with xyz coordinates, shape [B, N, 3].
            features: Tensor with features, shape [B, N, C].
        Output:
            Tensor with predictions, shape [B].
        """

        return self.head(self.backbone(xyz, features))

    def _shared_step(self, batch: torch.Tensor | tuple, batch_idx: int, prefix: str,
                     dataloader_idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Shared step between training and validation steps.

        Args:
            batch: Batch of data, tuple with (xyz, features, labels).
            batch_idx: Index of batch.
            prefix: Prefix for logging (train | val | test).
        
        Output:
            loss: Output from loss function.
            metrics: Dictionary with metrics, including loss.
        """

        if len(batch) == 3:
            # [B, N, 3], [B, N, C], [B]
            xyz, features, labels = batch
        elif len(batch) == 2:
            # [B, N, 3], [B]
            xyz, labels = batch
            features = None
        else:
            raise ValueError('Batch should be a tuple with (xyz, features, labels) or '
                             '(xyz, labels).')

        # pred [B, n_classes]
        pred = self.forward(xyz, features)
        loss = F.cross_entropy(pred, labels)

        pred_label = pred.argmax(dim=1)
        acc = (pred_label == labels).float().mean()

        metrics = {f'{prefix}_loss_{dataloader_idx}': loss, f'{prefix}_acc_{dataloader_idx}': acc}
        self.log(f'{prefix}_acc_{dataloader_idx}_',
                 acc.item(),
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss, metrics
