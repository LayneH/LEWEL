METHOD=LEWELB_EMAN
DSET='in1k'
LR=3.6
BS=1024
WD=0.0000015
BETA=0.98
EP=200
WARM_EP=10
LEWEL_SCALE=1.0
LEWEL_NUM_HEADS=4
LEWEL_LOSS_WEIGHT=0.5
PARTITION=$1
NTASKS=$2
PY_ARGS=${@:3}
JOB_NAME=${JOB_NAME:-"byol_eman_obj"}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
CKPT_DIR=./ckpts/${METHOD}/${DSET}/ep${EP}_${WARM_EP}_lr${LR}_bs${BS}_wd${WD}_m${BETA}_nh${LEWEL_NUM_HEADS}lw${LEWEL_LOSS_WEIGHT}s${LEWEL_SCALE}_${JOB_NAME}

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
        --save-dir ${CKPT_DIR} --save-freq 20 \
        ${PY_ARGS} \
        2>&1 | tee -a ${CKPT_DIR}/train-${NOW}.log &
