# LineEX: Data Extraction from Scientific Line Charts

This repo contains code and models for the LineEX system, (link paper), which extracts data from scientific line charts. We adapt existing vision transformers and pose detection methods and showcase significant performance gains over existing SOTA baselines. We also propose a new loss function and present its effectiveness against existing loss functions.

The LineEX pipeline consists of three modular stages, which can be used independent from each other. They are :

* Keypoint Extraction
* Chart Element Detection and Text Extraction
* Keypoint Grouping, Legend Mapping and Datapoint Scaling

## Usage

Install the dependencies:

```
python -m venv pyenv
source pyenv/bin/activate
pip install -r requirements.txt
```

Each of the modules can be used separately, or the entire pipeline can be called at once to extract the desired information.

### Keypoint Extraction

### Chart Element Detection and Text Extraction

Training:

`python -m torch.distributed.launch --nproc_per_node=4 --node_rank=0 main_charts_dist.py --coco_path path_to_data --batch_size 14 --dataset_file charts --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth`

The weights for the element detection model are availabel at: [link]

Evaluation:

`python modules/CE_detection/run.py`

### Overall Pipeline

`python pipeline.py`

Results are stored in `sample_output`

### Citation
