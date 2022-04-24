set -x 

METHOD=LEWELB
DSET='in1k'
LR=7.2
BS=4096
WD=0.000001
BETA=0.996
EP=400
WARM_EP=10
LEWEL_SCALE=1.0
LEWEL_NUM_HEADS=4
LEWEL_LOSS_WEIGHT=0.5
CKPT_DIR=./ckpts/${METHOD}/${DSET}/ep${EP}_${WARM_EP}_lr${LR}_bs${BS}_wd${WD}_m${BETA}_nh${LEWEL_NUM_HEADS}lw${LEWEL_LOSS_WEIGHT}s${LEWEL_SCALE}_${JOB_NAME}

PARTITION=$1
NTASKS=$2
PY_ARGS=${@:3}
JOB_NAME=${JOB_NAME:-"lewel"}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PYTHON=${PYTHON:-"python"}

mkdir -p ./ckpts
mkdir -p ${CKPT_DIR}
echo ${CKPT_DIR}
NOW=$(date +"%Y%m%d_%H%M%S")


GLOG_vmodule=MemcachedClient=-1 \
srun -p ${PARTITION} \
    --mpi=pmi2 --gres=gpu:${GPUS_PER_NODE} \
    -n${NTASKS} --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --job-name=${JOB_NAME} --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    ${PYTHON} -u main.py \
        --arch ${METHOD} --backbone resnet50_encoder \
        --dataset ${DSET} \
        --lr ${LR} -b ${BS} --wd ${WD} \
        --enc-m ${BETA} \
        --epochs ${EP} --cos --warmup-epoch ${WARM_EP} \
        --lewel-l2-norm \
        --lewel-scale ${LEWEL_SCALE} --lewel-num-heads ${LEWEL_NUM_HEADS} \
        --lewel-loss-weight ${LEWEL_LOSS_WEIGHT} \
        --amp \
        --norm SyncBN \
        --save-dir ${CKPT_DIR} --save-freq 20 \
        ${PY_ARGS} \
        2>&1 | tee -a ${CKPT_DIR}/train-${NOW}.log &
