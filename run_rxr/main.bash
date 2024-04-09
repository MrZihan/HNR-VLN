export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

flag1="--exp_name release_rxr
      --run-type train
      --exp-config run_rxr/iter_train.yaml
      SIMULATOR_GPU_IDS [0,1,2,3]
      TORCH_GPU_IDS [0,1,2,3]
      GPU_NUMBERS 4
      NUM_ENVIRONMENTS 1
      IL.iters 20000
      IL.lr 1e-5
      IL.log_every 500
      IL.ml_weight 1.0
      IL.sample_ratio 0.75
      IL.decay_interval 10000000
      IL.load_from_ckpt False
      IL.is_requeue True
      IL.waypoint_aug  True
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      IL.expert_policy ndtw
	  MODEL.pretrained_path data/ckpt.iter14800.pth
      "

flag2="--exp_name release_rxr
      --run-type eval
      --exp-config run_rxr/iter_train.yaml
      SIMULATOR_GPU_IDS [0,1]
      TORCH_GPU_IDS [0,1]
      GPU_NUMBERS 2
      NUM_ENVIRONMENTS 1
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING False
      EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_rxr/ckpt.iter4000.pth
      IL.back_algo control
      "

flag3="--exp_name release_rxr
      --run-type inference
      --exp-config run_rxr/iter_train.yaml
      SIMULATOR_GPU_IDS [4,5,6,7]
      TORCH_GPU_IDS [4,5,6,7]
      GPU_NUMBERS 4
      NUM_ENVIRONMENTS 8
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING False
      INFERENCE.CKPT_PATH data/logs/checkpoints/release_rxr/ckpt.iter22600.pth
      INFERENCE.PREDICTIONS_FILE preds.jsonl
      IL.back_algo control
      "

mode=$1
case $mode in 
      train)
      echo "###### train mode ######"
      CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --nproc_per_node=4 --master_port $2 run.py $flag1
      ;;
      eval)
      echo "###### eval mode ######"
      CUDA_VISIBLE_DEVICES='4,5' python -m torch.distributed.launch --nproc_per_node=2 --master_port $2 run.py $flag2
      ;;
      infer)
      echo "###### infer mode ######"
      CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --nproc_per_node=4 --master_port $2 run.py $flag3
      ;;
esac