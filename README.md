# Toward Reliable Photovoltaic Cell Inspection: A Robust Dual-Network Learning Framework for Electroluminescence Image Defect Classification
This is the official code for "Toward Reliable Photovoltaic Cell Inspection: A Robust Dual-Network Learning Framework for Electroluminescence Image Defect Classification".
## Dataset
The training and prediction is performed on orginal binary ELPV dataset and modified ELPV dataset-2024/2026.
The original dataset can be found on this [github page](https://github.com/zae-bayern/elpv-dataset). 
The modified dataset-2024/2026 can be found on [goggle drive](https://drive.google.com/drive/folders/1eNnqk4b13eyf9QBHO_oHbc2uZQ1OXhbL?usp=drive_link) or [baidu drive](https://pan.baidu.com/s/1WMBuDaJCvgnWfEm-ieF-Ag?pwd=1234). Moreover, you can download the orginal ELPV dataset and re-run the "make_modified_elpv2024.py"/"make_modified_elpv2026.py" to generate new datasets.

## Training

To train on the binary dataset, run the following command in the root path of this repository:

```shell
bash run.sh
# without .sh file
python train_merge.py --num_class 2 --backbone vgg16 --merge_t 5 --alpha 1. --lr 0.005 --gpuid 0
```

To train on the 4-class or 8-class dataset, run the following command in the the path of "./4_8_class_code/elpv2024" and "./4_8_class_code/elpv2026":

```shell
cd ./4_8_class_code/elpv2026
bash run.sh
```
