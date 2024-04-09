export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

flag1="--exp_name release_rxr
      --run-type train
      --exp-config run_rxr/iter_train_nerf.yaml
      SIMULATOR_GPU_IDS [7]
      TORCH_GPU_IDS [7]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 1
      IL.iters 40000
      IL.lr 2e-5
      IL.log_every 200
      IL.ml_weight 1.0
      IL.sample_ratio 0.75
      IL.decay_interval 1000000000
      IL.load_from_ckpt True
	  IL.ckpt_to_load data/logs/checkpoints/release_rxr/ckpt.iter14800.pth
      IL.is_requeue True
      IL.waypoint_aug  True
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      IL.expert_policy ndtw
      "
	  #MODEL.pretrained_path pretrained/ETP/mlm.sap_rxr/ckpts/model_step_90000.pt

flag2=" --exp_name release_rxr
      --run-type eval
      --exp-config run_rxr/iter_train_nerf.yaml
      SIMULATOR_GPU_IDS [7]
      TORCH_GPU_IDS [7]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 3
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING False
      EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_rxr/ckpt.iter31200.pth
      IL.back_algo control
      "

flag3="--exp_name release_rxr
      --run-type inference
      --exp-config run_rxr/iter_train_nerf.yaml
      SIMULATOR_GPU_IDS [0,1,2,3]
      TORCH_GPU_IDS [0,1,2,3]
      GPU_NUMBERS 4
      NUM_ENVIRONMENTS 8
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING False
      INFERENCE.CKPT_PATH data/logs/checkpoints/release_rxr/ckpt.iter19600.pth
      INFERENCE.PREDICTIONS_FILE preds.jsonl
      IL.back_algo control
      "

mode=$1
case $mode in 
      train)
      echo "###### train mode ######"
      python -m torch.distributed.launch --nproc_per_node=1 --master_port $2 run_nerf.py $flag1
      ;;
      eval)
      echo "###### eval mode ######"
      python -m torch.distributed.launch --nproc_per_node=1 --master_port $2 run_nerf.py $flag2
      ;;
      infer)
      echo "###### infer mode ######"
      python -m torch.distributed.launch --nproc_per_node=4 --master_port $2 run_nerf.py $flag3
      ;;
esac