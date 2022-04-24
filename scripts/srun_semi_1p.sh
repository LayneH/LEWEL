DSET='in1k'
LR=0.0001
LR_CLS=1.0
PARTITION=$1
NTASKS=$2
PRETRAINED=$3
JOB_NAME=${JOB_NAME:-"ft_1p"}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
CKPT_DIR=$(dirname $PRETRAINED)
CKPT_DIR=${CKPT_DIR}/semi_lr${LR}_LRC${LR_CLS}_${JOB_NAME}

mkdir -p ./ckpts
mkdir -p ${CKPT_DIR}
echo ${CKPT_DIR}
NOW=$(date +"%Y%m%d_%H%M%S")

PYTHON=${PYTHON:-"python"}

GLOG_vmodule=MemcachedClient=-1 \
srun -p ${PARTITION} \
    --mpi=pmi2 --gres=gpu:${GPUS_PER_NODE} \
    -n${NTASKS} --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --job-name=${JOB_NAME} --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    ${PYTHON} -u main_semi.py \
        -a resnet50 \
        --lr ${LR} --lr-classifier ${LR_CLS} \
        --epochs 50 --schedule 30 40 \
        --eval-freq 5 \
        --trainindex data/1percent.txt.ext \
        --model-prefix online_net.backbone \
        --self-pretrained ${PRETRAINED} \
        --save-dir ${CKPT_DIR} \
        2>&1 | tee -a ${CKPT_DIR}/train-${NOW}.log &