"""Configuration extracted from the pinned CFCD wrapper; paths are local."""
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parent/'vendor/CFCD'
sys.path.insert(0,str(REPO))
from core.config import cfg

def configure(args):
    cfg.defrost()
    cfg.merge_from_file(str(REPO / "configs/resnet50_cfcd_s2_8gpu.yaml"))
    cfg.NUM_GPUS = 1
    cfg.RNG_SEED = args.seed
    cfg.MODEL.ALGO = "MODEL_S2"
    cfg.MODEL.NUM_CLASSES = 277
    cfg.MODEL.HEADS.NUM_CLASSES = 277
    cfg.MODEL.HEADS.HEAD = "MadaCos"
    cfg.MODEL.HEADS.ALPHA = 0.02
    cfg.MODEL.HEADS.REDUCTION_DIM = 512
    cfg.MODEL.LOSSES.NAME = "triplet_loss"
    cfg.MODEL.LOSSES.LAMDA = 0.05
    cfg.MODEL.LOSSES.MARGIN = 0.1
    cfg.TRAIN.IM_SIZE = args.image_size
    cfg.TRAIN.BATCH_SIZE = args.batch_size
    cfg.TRAIN.WEIGHTS = str(args.s1_checkpoint.resolve())
    cfg.TRAIN.PRETRAINED = False
    cfg.TRAIN.AUTO_RESUME = True
    cfg.TRAIN.EVAL_PERIOD = args.val_period
    cfg.TRAIN.CHECKPOINT_PERIOD = args.val_period
    cfg.TEST.IM_SIZE = args.image_size
    cfg.DATA_LOADER.TUPLE_SIZE = 8
    cfg.DATA_LOADER.SAMPLER = "DistributedTupleHEMSampler"
    cfg.DATA_LOADER.NUM_WORKERS = args.workers
    cfg.OPTIM.BASE_LR = args.lr
    cfg.OPTIM.MAX_EPOCH = args.epochs
    cfg.OUT_DIR = str(args.output)
    cfg.LOG_DEST = "stdout"
    cfg.freeze()
