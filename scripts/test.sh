export CUDA_VISIBLE_DEVICES=0
eval "$(conda shell.bash hook)"
conda activate ml
python train.py --config-name=tinyshak \
    data.batch_size=16 \
    train.num_iter=10_000 \
    logging.log_freq=100 \
    logging.save_freq=1000