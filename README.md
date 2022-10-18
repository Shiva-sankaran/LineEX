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
```
chmod +x download.sh
./download.sh
```

## Testing
Each of the modules can be used separately, or the entire pipeline can be called at once to extract the desired information. Output is stored in the corresponding directory

### Overall
```
python pipeline.py
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

## Training (DATA NOT ADDED YET)



### Keypoint Extraction
```
cd modules/KP_detection
python train.py
```

### Chart Element Detection and Text Extraction
```
cd modules/CE_detection
python -m torch.distributed.launch train.py --coco_path path_to_data

```



## TBA
Need to change data paths

Add folders for data and weight 
## Citation
