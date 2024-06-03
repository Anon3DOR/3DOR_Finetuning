# conda deactivate
# conda activate 3DOR_2024

cd source/scripts/

python eval_linear.py --model ../../checkpoints/ModelNet40/checkpoints/last.state_dict --dataset ../../source/config/versioning/m40_v0.json --data_cache_dir ../../data_final/cache/M40/ --fold test
python eval_linear.py --model ../../checkpoints/ShapenetNormal/checkpoints/last.state_dict --dataset ../../source/config/versioning/shapenet_normal_coarse_v0.json --data_cache_dir ../../data_final/cache/shapenet_normal/ --fold test
python eval_linear.py --model ../../checkpoints/ShapenetPerturbed/checkpoints/last.state_dict --dataset ../../source/config/versioning/shapenet_perturbed_coarse_v0.json --data_cache_dir ../../data_final/cache/shapenet_perturbed/ --fold test
python eval_linear.py --model ../../checkpoints/MCB/checkpoints/last.state_dict --dataset ../../source/config/versioning/mcb_v0.json --data_cache_dir ../../data_final/cache/MCB/ --fold val
python eval_linear.py --model ../../checkpoints/obja_easy/checkpoints/last.state_dict --dataset ../../source/config/versioning/objaverse_easy_v0.json --data_cache_dir ../../data_final/cache/objaverse/ --fold test
python eval_linear.py --model ../../checkpoints/ScanObjectNN/checkpoints/last.state_dict --dataset ../../source/config/versioning/scanobjectnn_v0.json --data_cache_dir ../../data_final/cache/scanobjectnn/ --fold test

cd ../../