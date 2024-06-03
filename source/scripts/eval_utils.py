import os
import pathlib

import hydra
import numpy as np
import omegaconf
import torch
from sklearn import metrics as sk_metrics
from tqdm import tqdm

from source.models.ulip_pointbert.dvae import knn_point


def set_cuda_gpus(gpus: list[int]):
    """
    Takes a list of GPU ids and sets CUDA_VISIBLE_DEVICES accordingly.
    """

    if len(gpus) == 0:
        print('No gpus available in config. No GPUs will be set to CUDA_VISIBLE_DEVICES.')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus).replace(' ', '')[1:-1]


def load_model_from_checkpoint_path(checkpoint_path: str):

    checkpoint_path = pathlib.Path(checkpoint_path)
    checkpoint_dir = checkpoint_path.parent.parent
    checkpoint_config_path = checkpoint_dir / '.hydra' / 'config.yaml'
    checkpoint_config = omegaconf.OmegaConf.load(checkpoint_config_path)

    model = hydra.utils.instantiate(checkpoint_config.model, _recursive_=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['state_dict'])
    model.eval()

    return model, checkpoint_config


def load_model_from_config_path(config_path: str):

    config_path = pathlib.Path(config_path)
    config_dir = config_path.parent
    with hydra.initialize(config_path=str(config_dir), version_base='1.2'):
        cfg = hydra.compose(config_name=config_path.stem)

    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.eval()

    return model, cfg


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embedding_sizes = np.linalg.norm(embeddings, axis=1)
    return embeddings / embedding_sizes[:, np.newaxis]


def infer_embeddings(model: torch.nn.Module,
                     dataloader: torch.utils.data.DataLoader,
                     normals: bool = False,
                     full_model: bool = False,
                     normalize: bool = False):
    all_embeddings = []
    all_labels = []
    for batch in tqdm(dataloader):
        normals = None
        labels = None
        if len(batch) == 1:
            points = batch[0]
        if len(batch) == 2:
            if normals:
                points, normals = batch
                normals = normals.cuda()
            else:
                points, labels = batch
        else:
            points, normals, labels = batch
        points = points.cuda()
        with torch.no_grad():
            if full_model:
                all_embeddings.append(model(points, normals))
            else:
                all_embeddings.append(model.backbone(points, normals))
        all_labels.append(labels)
    all_embeddings = torch.vstack(all_embeddings).cpu().numpy()
    all_labels = torch.hstack(all_labels).numpy()
    if normalize:
        all_embeddings = normalize_embeddings(all_embeddings)
    return all_embeddings, all_labels


def _one_hot_encoding(arr: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes)[arr]


def calc_ndcg_score(train_emb: np.ndarray,
                    train_labels: np.ndarray,
                    test_emb: np.ndarray,
                    test_labels: np.ndarray,
                    k: int = 100):
    """
    Calculates the NDCG score for a given test set and train set.
    """

    test_retrieval = knn_point(
        k,
        torch.from_numpy(train_emb).unsqueeze(0).cuda(),
        torch.from_numpy(test_emb).unsqueeze(0).cuda(),
        sorted=False,
    ).cpu().squeeze(0).numpy()

    test_pred = train_labels[test_retrieval]
    test_target = test_labels.reshape(-1, 1)
    test_target = np.repeat(test_target, k, axis=1)

    num_classes = max([np.max(train_labels), np.max(test_labels)]) + 1
    one_hot_pred = _one_hot_encoding(test_pred, num_classes=num_classes)
    one_hot_target = _one_hot_encoding(test_target, num_classes=num_classes)
    results = []
    for pred, gt in tqdm(zip(one_hot_pred, one_hot_target)):
        results.append(sk_metrics.ndcg_score(gt, pred))
    return np.mean(results)
