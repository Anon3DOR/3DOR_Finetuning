import hydra
import omegaconf
import pytorch_lightning as pl
import torch


class ModelBase(pl.LightningModule):

    def __init__(
        self,
        backbone: omegaconf.DictConfig,
        head: omegaconf.DictConfig,
        optimizer: omegaconf.DictConfig,
        scheduler: omegaconf.DictConfig | None = None,
        backbone_pretrain_path: str | None = None,
        freeze_backbone: bool = False,
        tuning_metric: str | None = None,
        tuning_metric_mode: str | None = None,
        enable_knn_eval: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.backbone_cfg = backbone
        self.head_cfg = head

        # These are typically overwritten in the main config file and not in the model config.
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler
        self.backbone_pretrain_path = backbone_pretrain_path
        self.freeze_backbone = freeze_backbone

        self.backbone = hydra.utils.instantiate(self.backbone_cfg, _recursive_=False)
        self.head = hydra.utils.instantiate(self.head_cfg, _recursive_=False)

        if self.backbone_pretrain_path:
            self.load_backbone_state_dict()
            if self.freeze_backbone:
                self.backbone.requires_grad_(False)

        self.tuning_metric = tuning_metric
        self.tuning_metric_mode = tuning_metric_mode
        if tuning_metric and tuning_metric_mode:
            self.tuning_metric_top_3 = []

        self.enable_knn_eval = enable_knn_eval  # Expects val dataset to be class dataset.
        if enable_knn_eval:
            self.eval_embeddings = {}
            self.eval_labels = {}

    def load_backbone_state_dict(self):

        orig_state_dict = torch.load(self.backbone_pretrain_path,
                                     map_location=self.device)['state_dict']
        if len(orig_state_dict) == len(self.backbone.state_dict()):
            self.backbone.load_state_dict(orig_state_dict)
        else:
            backbone_state_dict = {
                k.replace('backbone.', ''): v
                for k, v in orig_state_dict.items()
                if k.startswith('backbone.')
            }
            self.backbone.load_state_dict(backbone_state_dict, strict=False)
        print('Loaded backbone state dict from ', self.backbone_pretrain_path)

    # For debugging purposes. Makes TensorboardLogger crash if input_feuture_dim is 0.
    # @property
    # def example_input_array(self) -> tuple[torch.Tensor, torch.Tensor]:
    #     """Allows tensorboard to log the computational graph."""

    #     if self.backbone.input_feature_dim != 0:
    #         return (torch.randn((2, 2048, 3), device=self.device),
    #                 torch.randn((2, 2048, self.backbone.input_feature_dim), device=self.device))
    #     else:
    #         return (torch.randn((2, 2048, 3), device=self.device), None)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            xyz: Tensor with xyz coordinates, shape [B, N, 3].
            features: Tensor with features, shape [B, N, C].

        Output:
            Tensor with predictions, shape [B, ?].
        """
        raise NotImplementedError(
            "You are in the ModelBase class where forward is not implemented.")

    def _shared_step(self,
                     batch: torch.Tensor | tuple,
                     batch_idx: int,
                     prefix: str,
                     dataloader_idx: int = 0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Shared step between training and validation steps.

        Args:
            batch: Batch of data.
            batch_idx: Index of batch.
            prefix: Prefix for logging (train | val | test).
        
        Output:
            loss: Output from loss function.
            metrics: Dictionary with metrics, including loss.
        """
        raise NotImplementedError(
            "You are in the ModelBase class where _shared_step is not implemented.")

    def training_step(self, batch: torch.Tensor | tuple,
                      batch_idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss, metrics = self._shared_step(batch, batch_idx, 'train', dataloader_idx=0)

        # TODO: Optimize logger settings for training speed?
        self.log_dict(metrics,
                      on_step=True,
                      on_epoch=True,
                      logger=True,
                      prog_bar=True,
                      rank_zero_only=True)
        # if self.tuning_metric and 'train' in self.tuning_metric:
        #     self.tuning_metric_accumulator.append(metrics[self.tuning_metric].detach())
        return loss

    def validation_step(self,
                        batch: torch.Tensor | tuple,
                        batch_idx: int,
                        dataloader_idx: int = 0) -> dict[str, torch.Tensor]:
        _, metrics = self._shared_step(batch, batch_idx, 'val', dataloader_idx)

        self.log_dict(metrics,
                      on_step=False,
                      on_epoch=True,
                      logger=True,
                      prog_bar=False,
                      rank_zero_only=True)
        # if self.tuning_metric and 'val' in self.tuning_metric and not self.trainer.sanity_checking and self.tuning_metric in metrics:
        #     self.tuning_metric_accumulator.append(metrics[self.tuning_metric].detach())

        if self.enable_knn_eval and not self.trainer.sanity_checking:
            if len(batch) == 2:
                xyz, label = batch
                features = None
            else:
                xyz, features, label = batch
            if str(dataloader_idx) not in self.eval_embeddings:
                self.eval_embeddings[str(dataloader_idx)] = []
                self.eval_labels[str(dataloader_idx)] = []
            self.eval_embeddings[str(dataloader_idx)].append(self.backbone(xyz, features))
            self.eval_labels[str(dataloader_idx)].append(label)

        return metrics

    def configure_optimizers(
            self) -> dict[str, torch.optim.Optimizer | torch.optim.lr_scheduler._LRScheduler]:

        optimizer = hydra.utils.instantiate(self.optimizer_cfg, params=self.parameters())
        return_dict = {'optimizer': optimizer}

        if self.scheduler_cfg is None:
            return return_dict

        scheduler = hydra.utils.instantiate(self.scheduler_cfg, optimizer=optimizer)
        return_dict['lr_scheduler'] = scheduler
        return return_dict

    # def on_train_epoch_end(self):
    #     if self.tuning_metric and 'train' in self.tuning_metric:
    #         self._log_tuning_metric()

    # def on_validation_epoch_end(self):
    #     if self.tuning_metric and 'val' in self.tuning_metric:
    #         self._log_tuning_metric()

    def _knn_eval(self, train_emb: torch.Tensor, train_labels: torch.Tensor, test_emb: torch.Tensor,
                  test_labels: torch.Tensor) -> float:

        dists = torch.cdist(test_emb, train_emb, p=2)
        neighbors = dists.argmin(dim=1)
        accuracy = (test_labels == train_labels[neighbors]).float().mean()

        return accuracy

    def on_validation_epoch_end(self):

        if self.enable_knn_eval and not self.trainer.sanity_checking:
            knn_scores = []

            for i in range(0, len(self.eval_embeddings), 2):
                train_emb = torch.vstack(self.eval_embeddings[str(i)])
                train_labels = torch.hstack(self.eval_labels[str(i)])
                test_emb = torch.vstack(self.eval_embeddings[str(i + 1)])
                test_labels = torch.hstack(self.eval_labels[str(i + 1)])

                knn_score = self._knn_eval(train_emb, train_labels, test_emb, test_labels)
                knn_scores.append(knn_score)
                self.eval_embeddings[str(i)].clear()
                self.eval_embeddings[str(i + 1)].clear()
                self.eval_labels[str(i)].clear()
                self.eval_labels[str(i + 1)].clear()
                self.log(f'val_knn_acc_{i}_{i + 1}', knn_score, logger=True)

            val_knn_acc_mean = torch.sum(torch.Tensor(knn_scores)) / len(knn_scores)
            self.log('val_knn_acc_mean', val_knn_acc_mean.item(), logger=True, prog_bar=True)

            if self.tuning_metric == 'val_knn_acc_mean':
                self._log_tuning_metric(val_knn_acc_mean)

    def _log_tuning_metric(self, metric_epoch_result):
        """Logs the top3 result for a metric during hyperparameter tuning."""

        if len(self.tuning_metric_top_3) < 3:
            self.tuning_metric_top_3.append(metric_epoch_result)
        else:
            index_to_replace = -1
            if self.tuning_metric_mode == 'min':
                worst_in_buffer = max(self.tuning_metric_top_3)
                if metric_epoch_result < worst_in_buffer:
                    index_to_replace = self.tuning_metric_top_3.index(worst_in_buffer)
            elif self.tuning_metric_mode == 'max':
                worst_in_buffer = min(self.tuning_metric_top_3)
                if metric_epoch_result > worst_in_buffer:
                    index_to_replace = self.tuning_metric_top_3.index(worst_in_buffer)

            if index_to_replace != -1:
                self.tuning_metric_top_3[index_to_replace] = metric_epoch_result

        final_result = torch.stack(self.tuning_metric_top_3).mean()
        self.log(f'{self.tuning_metric}_top3', final_result, on_step=False, on_epoch=True)
