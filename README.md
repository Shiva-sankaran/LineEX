# LineEX: Data Extraction from Scientific Line Charts

This repo contains code and models for the LineEX system, (link paper), which extracts data from scientific line charts. We adapt existing vision transformers and pose detection methods and showcase significant performance gains over existing SOTA baselines. We also propose a new loss function and present its effectiveness against existing loss functions.

The LineEX pipeline consists of three modular stages, which can be used independent from each other. They are :

* Keypoint Extraction
* Chart Element Detection and Text Extraction
* Keypoint Grouping, Legend Mapping and Datapoint Scaling

## Usage

### Clone this repository:
```
git clone https://github.com/Shiva-sankaran/LineEX.git
cd LineEX
```
### Install the dependencies:

```
conda env create -f environment.yml
conda activate LineEX
```

### Download weights and data
Weights and data will be placed at the correct folders

Set corresponding DATA_flag(True/False) to download a particular data set.

```
chmod +x download.sh
./download.sh -T False -V False  -L True  # To download only the test data 
```

### UPDATE: Dataset moved to [here](https://iitgnacin-my.sharepoint.com/:f:/g/personal/md_hassan_iitgn_ac_in/EnX4sNoMnrdAmCVEB55r95EB_h5Xa_uk04zvEPg5ZLLGZw?e=KAILBD).
### UPDATE: Weights can be found [here](https://drive.google.com/drive/folders/15bKREf2EBORHBZx_xz53eVJzs9e4cZ7Q?usp=sharing)

## Testing
Each of the modules can be used separately, or the entire pipeline can be called at once to extract the desired information. Output is stored in the corresponding directory

### Overall
```
python pipeline.py --input_path = sample_input/
```
### Keypoint detection
```
cd modules/KP_detection
python run.py
```
### Chart element detection
```
cd modules/CE_detection
python run.py
```

## Evaluation
Refer to the paper for more information about the metrics

### Overall

Overall metrics is essentially the metric for grouping and legend mapping
```
cd modules/Grouping_legend_mapping
python eval.py
```
### Keypoint detection
```
cd modules/KP_detection
python eval.py
```
### Chart element detection
```
cd modules/CE_detection
python run.py
```

## Training 



### Keypoint Extraction
```
cd modules/KP_detection
python -m torch.distributed.launch --nproc_per_node=3 --node_rank=0 train.py --vit_arch xcit_small_12_p16 --batch_size 42 --input_size 288 384 --hidden_dim 384 --vit_dim 384 --num_workers 24 --vit_weights https://dl.fbaipublicfiles.com/xcit/xcit_small_12_p16_384_dist.pth --alpha 0.99
```

### Chart Element Detection and Text Extraction
```
cd modules/CE_detection
python -m torch.distributed.launch train.py --coco_path path_to_data

```



## TBA
Need to change data paths

## Citation

Shivasankaran, V. P., Muhammad Yusuf Hassan, and Mayank Singh. "LineEX: Data Extraction from Scientific Line Charts." 2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 2023.
