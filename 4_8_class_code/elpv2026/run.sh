#run the 4-class elpv2026 dataset experiments
python train_merge.py --num_class 4 --backbone vgg16 --merge_t 5 --alpha 1. --lr 0.005 --gpuid 0

#run the 8-class elpv2026 dataset experiments
python train_merge.py --num_class 8 --backbone vgg16 --merge_t 5 --alpha 1. --lr 0.005 --gpuid 0

