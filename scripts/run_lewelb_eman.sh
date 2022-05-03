METHOD=LEWELB_EMAN
DSET='im_folder'
DATA_ROOT="PATH/TO/IMAGENET"
LR=3.6
BS=1024
WD=0.0000015
BETA=0.98
EP=200
WARM_EP=10
LEWEL_SCALE=1.0
LEWEL_NUM_HEADS=4
LEWEL_LOSS_WEIGHT=0.5
PY_ARGS=${@:1}
PORT=${PORT:-5389}
NPROC=${NPROC:-8}
CKPT_DIR=./ckpts/${METHOD}/${DSET}/ep${EP}_${WARM_EP}_lr${LR}_bs${BS}_wd${WD}_m${BETA}_nh${LEWEL_NUM_HEADS}lw${LEWEL_LOSS_WEIGHT}s${LEWEL_SCALE}_${JOB_NAME}

mkdir -p ./ckpts
mkdir -p ${CKPT_DIR}
echo ${CKPT_DIR}
NOW=$(date +"%Y%m%d_%H%M%S")

PYTHON=${PYTHON:-"python"}

${PYTHON} -u -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NPROC} \
    main.py \
    --arch ${METHOD} --backbone resnet50_encoder \
    --dataset ${DSET} \
    --data-root ${DATA_ROOT} \
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
