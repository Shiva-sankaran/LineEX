echo "Downloading weights"
gdown 176BjH_6W-HRoU9RvsysSNW27GVIHPlHV
mkdir temp/
mkdir data/
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


 while getopts "T:V:L:" flag
    do
             case "${flag}" in
                    T) TRAIN=${OPTARG};;
                    V) VAL=${OPTARG};;
                    L) TEST=${OPTARG};;
                    # I) TEST_PROJECTID=${OPTARG};;
             esac
    done
    echo "TRAIN: $TEST_USER";
    echo "VAL: $TEST_PWD";
    echo "TEST: $TEST_JOBID";

if [ $TRAIN == "True"] || [ $VAL == "True" ] || [ $TEST == "True" ]
then 
   mkdir data/
fi

if [ $TRAIN == "True" ]
then
   echo "Downloading Train Data"
   gdown 1x-2A0TXTY2cL7390NSvVXrt9skIuUKkR
   mkdir data/train
   unzip -d temp/ train.zip
   mv temp/train/anno data/train/
   mv temp/train/images data/train/


fi

if [ $VAL == "True" ]
then
   echo "Downloading Val Data"
   gdown 1wr1pezZ3teMiS3k4TCNcNvH7TigtiGMF
   mkdir data/val
   unzip -d temp/ val.zip
   mv temp/val/anno data/val/
   mv temp/val/images data/val/
fi

if [ $TEST == "True" ]
then
   echo "Downloading Test Data"
   gdown 1nDSGrYqUrwBg-8YkcvWLmGU5mtyfA1uz
   mkdir data/test
   unzip -d temp/ test.zip
   mv temp/test/anno data/test/
   mv temp/test/images data/test/
fi
    
rm -rf temp/
    
    
    # echo "PROJECTID: $TEST_PROJECTID";

# https://drive.google.com/file/d/176BjH_6W-HRoU9RvsysSNW27GVIHPlHV/view?usp=sharing