# NEED TO ADD DATA
echo "Downloading weights"
gdown 176BjH_6W-HRoU9RvsysSNW27GVIHPlHV
mkdir temp/
mkdir modules/CE_detection/ckpts
mkdir modules/KP_detection/ckpts
mkdir modules/Grouping_legend_mapping/ckpts
echo "Extracting checkpoints"
unzip -d temp/ weights.zip
mv temp/weights/checkpoint110.pth modules/CE_detection/ckpts/1.pth
mv temp/weights/ckpt_30.t7 modules/Grouping_legend_mapping/ckpts/1.t7
mv temp/weights/mlp_ckpt.t7 modules/Grouping_legend_mapping/ckpts/2.t7
mv temp/weights/ckpt_L.t7 modules/KP_detection/ckpts/1.t7
mv temp/weights/ckpt_L+D.t7 modules/KP_detection/ckpts/2.t7
echo "Moved checkpoints to corresponding dir"
rm -rf temp/

# https://drive.google.com/file/d/176BjH_6W-HRoU9RvsysSNW27GVIHPlHV/view?usp=sharing