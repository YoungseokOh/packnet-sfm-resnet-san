"""Default packnet_sfm configuration parameters (overridable in configs/*.yaml)
"""
import os
from yacs.config import CfgNode as CN

########################################################################################################################
cfg = CN()
cfg.name = ''       # Run name
cfg.debug = True   # Debugging flag

########################################################################################################################
### MODEL
########################################################################################################################
cfg.model = CN()
cfg.model.name = ''                         # Training model
cfg.model.checkpoint_path = ''              # Checkpoint path for model saving

########################################################################################################################
### MODEL.LOSS
########################################################################################################################
cfg.model.loss = CN()
# Rotation mode
cfg.model.loss.rotation_mode = 'euler'         # Rotation mode
cfg.model.loss.upsample_depth_maps = True      # Resize depth maps to highest resolution
#
cfg.model.loss.ssim_loss_weight = 0.85         # SSIM loss weight
cfg.model.loss.occ_reg_weight = 0.1            # Occlusion regularizer loss weight
cfg.model.loss.smooth_loss_weight = 0.001      # Smoothness loss weight
cfg.model.loss.C1 = 1e-4                       # SSIM parameter
cfg.model.loss.C2 = 9e-4                       # SSIM parameter
cfg.model.loss.photometric_reduce_op = 'min'   # Method for photometric loss reducing
cfg.model.loss.disp_norm = True                # Inverse depth normalization
cfg.model.loss.clip_loss = 0.0                 # Clip loss threshold variance
cfg.model.loss.padding_mode = 'zeros'          # Photometric loss padding mode
cfg.model.loss.automask_loss = True            # Automasking to remove static pixels
#
cfg.model.loss.velocity_loss_weight = 0.1      # Velocity supervision loss weight
#
cfg.model.loss.supervised_method = 'sparse-l1' # Method for depth supervision
cfg.model.loss.supervised_num_scales = 4       # Number of scales for supervised learning
cfg.model.loss.supervised_loss_weight = 0.9    # Supervised loss weight
cfg.model.loss.consistency_loss_weight = 0.1   # Consistency loss weight (for Yolov8 Semi-Supervised Learning)
# Scale-Adaptive Loss defaults
cfg.model.loss.lambda_sg = 0.5                 # Gradient loss weight for scale-adaptive loss
cfg.model.loss.num_scales = 4                  # Multi-scale gradient levels for scale-adaptive loss
cfg.model.loss.use_absolute = True             # Include absolute term in scale-adaptive loss
cfg.model.loss.use_inv_depth = False           # Operate directly on inverse depth in scale-adaptive loss
cfg.model.loss.epsilon = 1e-8                  # Numerical stability constant for scale-adaptive loss
# 🆕 Adaptive Multi-Domain Loss parameters
cfg.model.loss.ssi_weight = 0.7                # SSI loss weight in SSISilogLoss (for adaptive multi-domain)
cfg.model.loss.silog_weight = 0.3              # Silog loss weight in SSISilogLoss (for adaptive multi-domain)
cfg.model.loss.alpha_ssi = 0.85                # Alpha parameter for SSI loss (for adaptive multi-domain)
cfg.model.loss.beta_silog = 0.15               # Beta parameter for Silog loss (for adaptive multi-domain)
cfg.model.loss.min_depth = 0.05                # Minimum depth for clamping (for adaptive multi-domain)
cfg.model.loss.max_depth = 100.0               # Maximum depth for clamping (for adaptive multi-domain)
# 🆕 Fixed Multi-Domain Loss parameters (Phase 1)
cfg.model.loss.w_structure = 0.4               # Structure (gradient) loss weight in fixed multi-domain loss
cfg.model.loss.w_scale = 0.6                   # Scale (absolute depth) loss weight in fixed multi-domain loss
cfg.model.loss.alpha = 0.85                    # Alpha parameter for SSISilogLoss (fixed multi-domain)
cfg.model.loss.silog_ratio = 10                # Silog ratio parameter (fixed multi-domain)
cfg.model.loss.silog_ratio2 = 0.85             # Silog ratio2 parameter (fixed multi-domain)
# 🆕 Near-field weighting for SSI-Silog loss (Phase 2)
cfg.model.loss.enable_near_field_weighting = False  # Enable distance-based pixel weighting for near-field prioritization
# 🛣️ Road-aware weighting for SSI-Silog loss (Autonomous Driving)
cfg.model.loss.enable_road_weighting = False   # Enable road-aware pixel weighting
cfg.model.loss.near_field_threshold = 1.0      # Near-field distance threshold (m)
cfg.model.loss.road_weight = 5.0               # Weight for road pixels
cfg.model.loss.road_nearfield_weight = 10.0    # Weight for road + near-field pixels
cfg.model.loss.nonroad_nearfield_weight = 3.0  # Weight for non-road near-field pixels
########################################################################################################################
### MODEL.DEPTH_NET
########################################################################################################################
cfg.model.depth_net = CN()
cfg.model.depth_net.name = ''               # Depth network name
cfg.model.depth_net.checkpoint_path = ''    # Depth checkpoint filepath
cfg.model.depth_net.version = ''            # Depth network version
cfg.model.depth_net.dropout = 0.0           # Depth network dropout
cfg.model.depth_net.force_output_shape = () # 🆕 출력 해상도 강제. 예: (384, 640)

# 🆕 SAN 관련 설정 추가 (기존 네트워크와 호환)
cfg.model.depth_net.use_film = False        # Enable Depth-aware FiLM
cfg.model.depth_net.film_scales = [0]       # Which scales to apply FiLM
cfg.model.depth_net.use_enhanced_lidar = False  # Enable enhanced LiDAR processing

# 🆕 ST2 Dual-Head 설정 (INT8 quantization 지원)
cfg.model.depth_net.use_dual_head = False   # Enable Dual-Head architecture (integer + fractional)

# 🆕 ReZero 관련 설정 (ResNetSAN01_ReZero 전용)
cfg.model.depth_net.use_encoder_rezero = False  # Enable ReZero in ResNet encoder blocks

# 🆕 YOLOv8SAN01 전용 파라미터 (선택적으로만 사용)
cfg.model.depth_net.variant = 's'           # YOLOv8 variant (n, s, m, l, x) - YOLOv8SAN01에서만 사용
cfg.model.depth_net.use_neck_features = False
cfg.model.depth_net.use_imagenet_pretrained = False  # Enable enhanced LiDAR processing in YOLOv8SAN01
cfg.model.depth_net.use_depth_neck = False
########################################################################################################################
### MODEL.POSE_NET
########################################################################################################################
cfg.model.pose_net = CN()
cfg.model.pose_net.name = ''                # Pose network name
cfg.model.pose_net.checkpoint_path = ''     # Pose checkpoint filepath
cfg.model.pose_net.version = ''             # Pose network version
cfg.model.pose_net.dropout = 0.0            # Pose network dropout

########################################################################################################################
### MODEL.OPTIMIZER
########################################################################################################################
cfg.model.optimizer = CN()
cfg.model.optimizer.name = 'Adam'               # Optimizer name
cfg.model.optimizer.depth = CN()
cfg.model.optimizer.depth.lr = 0.0002           # Depth learning rate
cfg.model.optimizer.depth.weight_decay = 0.0    # Depth weight decay
cfg.model.optimizer.pose = CN()
cfg.model.optimizer.pose.lr = 0.0002            # Pose learning rate
cfg.model.optimizer.pose.weight_decay = 0.0     # Pose weight decay

########################################################################################################################
### MODEL.SCHEDULER
########################################################################################################################
cfg.model.scheduler = CN()
cfg.model.scheduler.name = 'StepLR'     # Scheduler name
cfg.model.scheduler.step_size = 10      # Scheduler step size
cfg.model.scheduler.gamma = 0.5         # Scheduler gamma value
cfg.model.scheduler.T_max = 20

########################################################################################################################
### MODEL.PARAMS
########################################################################################################################
cfg.model.params = CN()
cfg.model.params.crop = 'garg'              # Crop type
cfg.model.params.min_depth = 0.0            # Minimum depth value
cfg.model.params.max_depth = 100.0          # Maximum depth value
cfg.model.params.scale_output = ''          # Scale output type
cfg.model.params.use_log_space = False    # 🆕 Use log space for depth representation
########################################################################################################################
### ARCH
########################################################################################################################
cfg.arch = CN()
cfg.arch.seed = 42                      # Random seed for Pytorch/Numpy initialization
cfg.arch.min_epochs = 1                 # Minimum number of epochs
cfg.arch.max_epochs = 50                # Maximum number of epochs
cfg.arch.validate_first = False         # Validate before training starts
cfg.arch.eval_during_training = True    # Enable evaluation during training
cfg.arch.eval_progress_interval = 0.1   # 10%마다 평가
cfg.arch.eval_subset_size = 25          # 25개 이미지만 평가
cfg.arch.clip_grad = 10.0               # Gradient clipping value
cfg.arch.dtype = None                   # Data type for training (None for default)

########################################################################################################################
### DATASETS
########################################################################################################################
cfg.datasets = CN()

########################################################################################################################
### DATASETS.AUGMENTATION
########################################################################################################################
cfg.datasets.augmentation = CN()
cfg.datasets.augmentation.image_shape = ()                      # Image shape
cfg.datasets.augmentation.jittering = (0.2, 0.2, 0.2, 0.05)     # Color jittering values
cfg.datasets.augmentation.crop_train_borders = ()               # Crop training borders
cfg.datasets.augmentation.crop_eval_borders = ()                # Crop evaluation borders

# 🆕 Advanced augmentation 설정 추가
cfg.datasets.augmentation.randaugment = CN()
cfg.datasets.augmentation.randaugment.enabled = False           # Enable RandAugment
cfg.datasets.augmentation.randaugment.n = 9                     # Number of augmentation operations
cfg.datasets.augmentation.randaugment.m = 0.5                   # Magnitude (0-1)
cfg.datasets.augmentation.randaugment.prob = 0.5                # Probability of applying RandAugment

cfg.datasets.augmentation.random_erasing = CN()
cfg.datasets.augmentation.random_erasing.enabled = False        # Enable RandomErasing
cfg.datasets.augmentation.random_erasing.probability = 0.1      # Probability of applying RandomErasing
cfg.datasets.augmentation.random_erasing.sl = 0.02              # Minimum erased area ratio
cfg.datasets.augmentation.random_erasing.sh = 0.4               # Maximum erased area ratio
cfg.datasets.augmentation.random_erasing.r1 = 0.3               # Minimum aspect ratio
cfg.datasets.augmentation.random_erasing.mean = [0.485, 0.456, 0.406]  # Fill values for erased regions

cfg.datasets.augmentation.mixup = CN()
cfg.datasets.augmentation.mixup.enabled = False                 # Enable MixUp
cfg.datasets.augmentation.mixup.alpha = 0.2                     # Beta distribution parameter
cfg.datasets.augmentation.mixup.prob = 0.5                      # Probability of applying MixUp

cfg.datasets.augmentation.cutmix = CN()
cfg.datasets.augmentation.cutmix.enabled = False                # Enable CutMix
cfg.datasets.augmentation.cutmix.alpha = 1.0                    # Beta distribution parameter
cfg.datasets.augmentation.cutmix.prob = 0.5                     # Probability of applying CutMix

########################################################################################################################
### DATASETS.TRAIN
########################################################################################################################
cfg.datasets.train = CN()
cfg.datasets.train.batch_size = 2                   # Training batch size
cfg.datasets.train.num_workers = 16                 # Training number of workers
cfg.datasets.train.back_context = 1                 # Training backward context
cfg.datasets.train.forward_context = 1              # Training forward context
cfg.datasets.train.dataset = []                     # Training dataset
cfg.datasets.train.path = []                        # Training data path
cfg.datasets.train.split = []                       # Training split
cfg.datasets.train.depth_type = ['']                # Training depth type
cfg.datasets.train.input_depth_type = ['']          # Training input depth type
cfg.datasets.train.cameras = [[]]                   # Training cameras (double list, one for each dataset)
cfg.datasets.train.repeat = [1]                     # Number of times training dataset is repeated per epoch
cfg.datasets.train.num_logs = 5                     # Number of training images to log
cfg.datasets.train.mask_file = ['']
cfg.datasets.train.use_mask = [False]  # ← list로 변경
########################################################################################################################
### DATASETS.VALIDATION
########################################################################################################################
cfg.datasets.validation = CN()
cfg.datasets.validation.batch_size = 1              # Validation batch size
cfg.datasets.validation.num_workers = 8             # Validation number of workers
cfg.datasets.validation.back_context = 0            # Validation backward context
cfg.datasets.validation.forward_context = 0         # Validation forward context
cfg.datasets.validation.dataset = []                # Validation dataset
cfg.datasets.validation.path = []                   # Validation data path
cfg.datasets.validation.split = []                  # Validation split
cfg.datasets.validation.depth_type = ['']           # Validation depth type
cfg.datasets.validation.input_depth_type = ['']     # Validation input depth type
cfg.datasets.validation.cameras = [[]]              # Validation cameras (double list, one for each dataset)
cfg.datasets.validation.num_logs = 5                # Number of validation images to log
cfg.datasets.validation.mask_file = ['']
cfg.datasets.validation.use_mask = [False]  # ← list로 변경
########################################################################################################################
### DATASETS.TEST
########################################################################################################################
cfg.datasets.test = CN()
cfg.datasets.test.batch_size = 1                    # Test batch size
cfg.datasets.test.num_workers = 8                   # Test number of workers
cfg.datasets.test.back_context = 0                  # Test backward context
cfg.datasets.test.forward_context = 0               # Test forward context
cfg.datasets.test.dataset = []                      # Test dataset
cfg.datasets.test.path = []                         # Test data path
cfg.datasets.test.split = []                        # Test split
cfg.datasets.test.depth_type = ['']                 # Test depth type
cfg.datasets.test.input_depth_type = ['']           # Test input depth type
cfg.datasets.test.cameras = [[]]                    # Test cameras (double list, one for each dataset)
cfg.datasets.test.num_logs = 5                      # Number of test images to log
cfg.datasets.test.mask_file = ['']
cfg.datasets.test.use_mask = [False]  # ← list로 변경
########################################################################################################################
### CHECKPOINT
########################################################################################################################
cfg.checkpoint = CN()
cfg.checkpoint.filepath = ''            # Checkpoint filepath to save data
cfg.checkpoint.save_top_k = 5           # Number of best models to save
cfg.checkpoint.monitor = 'loss'         # Metric to monitor for logging
cfg.checkpoint.monitor_index = 0        # Dataset index for the metric to monitor
cfg.checkpoint.mode = 'auto'            # Automatically determine direction of improvement (increase or decrease)
cfg.checkpoint.period = 1               # Save checkpoint every N epochs
cfg.checkpoint.s3_path = ''             # s3 path for AWS model syncing
cfg.checkpoint.s3_frequency = 1         # How often to s3 sync

########################################################################################################################
### SAVE
########################################################################################################################
cfg.save = CN()
cfg.save.folder = ''                    # Folder where data will be saved
cfg.save.depth = CN()
cfg.save.depth.rgb = True               # Flag for saving rgb images
cfg.save.depth.viz = True               # Flag for saving inverse depth map visualization
cfg.save.depth.npz = True               # Flag for saving numpy depth maps
cfg.save.depth.png = True               # Flag for saving png depth maps

########################################################################################################################
### WANDB
########################################################################################################################
cfg.wandb = CN()
cfg.wandb.dry_run = True                                 # Wandb dry-run (not logging)
cfg.wandb.name = ''                                      # Wandb run name
cfg.wandb.project = os.environ.get("WANDB_PROJECT", "")  # Wandb project
cfg.wandb.entity = os.environ.get("WANDB_ENTITY", "")    # Wandb entity
cfg.wandb.tags = []                                      # Wandb tags
cfg.wandb.dir = ''                                       # Wandb save folder

########################################################################################################################
### TENSORBOARD
########################################################################################################################
cfg.tensorboard = CN()
cfg.tensorboard.dry_run = True                           # Tensorboard dry-run (not logging)
cfg.tensorboard.log_frequency = 100                      # How often to log images/depth maps (steps)
cfg.tensorboard.log_dir = ''                             # Tensorboard log directory

########################################################################################################################
### THESE SHOULD NOT BE CHANGED
########################################################################################################################
cfg.config = ''                 # Run configuration file
cfg.default = ''                # Run default configuration file
cfg.wandb.url = ''              # Wandb URL
cfg.checkpoint.s3_url = ''      # s3 URL
cfg.save.pretrained = ''        # Pretrained checkpoint
cfg.prepared = False            # Prepared flag

########################################################################################################################

def get_cfg_defaults():
    return cfg.clone()