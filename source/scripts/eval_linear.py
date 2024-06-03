import argparse
import json
import os
import pathlib

from sklearn import metrics as sk_metrics
from sklearn import neighbors as sk_neighbors


def set_cuda_gpus(gpus: list[int]):
    """
    Takes a list of GPU ids and sets CUDA_VISIBLE_DEVICES accordingly.
    """

    if len(gpus) == 0:
        print('No gpus available in config. No GPUs will be set to CUDA_VISIBLE_DEVICES.')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus).replace(' ', '')[1:-1]


def main(
    checkpoint_path: str,
    dataset_path: str,
    data_cache_dir: str,
    limit: int,
    fold: str,
    batch_size: int,
):
    import torch

    from source.datasets import dataset as source_dataset
    from source.scripts import eval_utils

    # Load model and dataset.
    model, config = eval_utils.load_model_from_checkpoint_path(checkpoint_path)
    model.cuda()

    with open(dataset_path, 'r', encoding='utf-8') as f:
        versioning = json.load(f)
    if fold == 'test':
        eval_fold = 'test'
        train_folds = ['train', 'val']
    elif fold == 'val':
        eval_fold = 'val'
        train_folds = ['train']
    else:
        eval_fold = 'train'
        train_folds = ['train']
    eval_dataset = source_dataset.ClassificationDataset(
        versioning_path=dataset_path,
        cache_folder_path=data_cache_dir,
        fold_mapping={'test': [eval_fold]},
        num_points=config.dataset.num_points,
        normals=config.dataset.normals,
        limit=limit,
        mode='test',
        transforms=None,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
    )
    train_dataset = source_dataset.ClassificationDataset(
        versioning_path=dataset_path,
        cache_folder_path=data_cache_dir,
        fold_mapping={'test': train_folds},
        num_points=config.dataset.num_points,
        normals=config.dataset.normals,
        limit=limit,
        mode='test',
        transforms=None,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
    )

    # Infer all embeddings for fold in given dataset.
    print('Infer embeddings for train split.')
    train_embeddings, train_labels = eval_utils.infer_embeddings(
        model,
        train_dataloader,
        config.dataset.normals,
        full_model=False,
    )
    print('Infer embeddings for eval split.')
    eval_embeddings, eval_labels = eval_utils.infer_embeddings(
        model,
        eval_dataloader,
        config.dataset.normals,
        full_model=False,
    )

    # Calculate acc.
    knn_model = sk_neighbors.KNeighborsClassifier(n_neighbors=1, n_jobs=10)
    knn_model.fit(train_embeddings, train_labels)
    eval_pred_knn = knn_model.predict(eval_embeddings)
    acc_score_knn = sk_metrics.accuracy_score(eval_labels, eval_pred_knn)
    f1_score_knn = sk_metrics.f1_score(eval_labels, eval_pred_knn, average='macro')
    acc_score_name = f'acc_knn_{versioning["name"]}_{eval_fold}'
    f1_score_name = f'f1_knn_{versioning["name"]}_{eval_fold}'
    print(f'{acc_score_name}:\n {acc_score_knn*100:.2f}')
    print(f'{f1_score_name}:\n {f1_score_knn*100:.2f}')

    # Calculate NDCG.
    ndcg_score = eval_utils.calc_ndcg_score(
        train_embeddings,
        train_labels,
        eval_embeddings,
        eval_labels,
        k=100,
    )
    ndcg_score_name = f'ndcg_{versioning["name"]}_{eval_fold}'
    print(f'{ndcg_score_name}:\n {ndcg_score*100:.2f}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate retrieval')
    parser.add_argument('--model', type=str, help='Model checkpoint path.')
    parser.add_argument('--dataset', type=str, help='Dataset versioning config path.')
    parser.add_argument('--data_cache_dir', type=str, help='Dataset cache directory.')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples.')
    parser.add_argument('--fold', type=str, default='test', help='Dataset fold.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for inference.')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID.')
    args = parser.parse_args()

    set_cuda_gpus([args.gpu_id])

    main(str(pathlib.Path(args.model).resolve()), str(pathlib.Path(args.dataset).resolve()), str(pathlib.Path(args.data_cache_dir).resolve()), args.limit, args.fold, args.batch_size)
