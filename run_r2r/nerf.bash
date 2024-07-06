export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

flag1="--exp_name release_r2r
      --run-type train
      --exp-config run_r2r/iter_train_nerf.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 1
      IL.iters 20000
      IL.lr 1e-4
      IL.log_every 1000
      IL.ml_weight 100.0
      IL.sample_ratio 0.75
      IL.decay_interval 10000000
      IL.load_from_ckpt True
      IL.is_requeue False
      IL.waypoint_aug  True
	  IL.ckpt_to_load data/NeRF_p16_8x8.pth
	  MODEL.pretrained_path data/ckpt.iter15000.pth
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      "

flag2=" --exp_name release_r2r
      --run-type eval
      --exp-config run_r2r/iter_train_nerf.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 1
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
	  MODEL.pretrained_path data/ckpt.iter15000.pth
      EVAL.CKPT_PATH_DIR data/NeRF_p16_8x8.pth
      IL.back_algo control
      "

flag3="--exp_name release_r2r
      --run-type inference
      --exp-config run_r2r/iter_train_nerf.yaml
      SIMULATOR_GPU_IDS [0,1]
      TORCH_GPU_IDS [0,1]
      GPU_NUMBERS 2
      NUM_ENVIRONMENTS 8
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      INFERENCE.CKPT_PATH data/logs/checkpoints/release_r2r/ckpt.iter12000.pth
      INFERENCE.PREDICTIONS_FILE preds.json
      IL.back_algo control
      "

mode=$1
case $mode in 
      train)
      echo "###### train mode ######"
      CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch --nproc_per_node=1 --master_port $2 run_nerf.py $flag1
      ;;
      eval)
      echo "###### eval mode ######"
      CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch --nproc_per_node=1 --master_port $2 run_nerf.py $flag2
      ;;
      infer)
      echo "###### infer mode ######"
      python -m torch.distributed.launch --nproc_per_node=2 --master_port $2 run_nerf.py $flag3
      ;;
esac