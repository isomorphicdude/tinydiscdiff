export CUDA_VISIBLE_DEVICES=0
eval "$(conda shell.bash hook)"
conda activate ml

############### Tiny Shakespeare Test ##################
# python train.py --config-name=tinyshak \
#     data.batch_size=16 \
#     train.num_iter=10_000 \
#     logging.log_freq=100 \
#     logging.save_freq=1000

############### MNIST Test ##################
# python train.py --config-name=mnist \
#     data.batch_size=16 \
#     train.num_iter=10000 \
#     logging.log_freq=100 \
#     logging.save_freq=500