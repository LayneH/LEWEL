DSET='in1k'
LR=0.4
BS=256
PARTITION=$1
NTASKS=$2
PRETRAINED=$3
JOB_NAME=${JOB_NAME:-"ev_lewel"}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
CKPT_DIR=$(dirname $PRETRAINED)
CKPT_DIR=${CKPT_DIR}/lincls_lr${LR}_${JOB_NAME}

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
    ${PYTHON} -u main_lincls.py \
        -a resnet50 \
        --dataset ${DSET} \
        --lr ${LR} \
        --epochs 50 --schedule 30 40 \
        --eval-freq 5 \
        --model-prefix online_net.backbone \
        --pretrained ${PRETRAINED} \
        --save-dir ${CKPT_DIR} \
        2>&1 | tee -a ${CKPT_DIR}/train-${NOW}.log &
