#run the binary dataset experiments
python train_merge.py --num_class 2 --backbone vgg16 --merge_t 5 --alpha 1. --lr 0.005 --gpuid 0
