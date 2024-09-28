#/bin/bash
#@author:meetdoshi 
#Courtest of https://github.com/AI4Bharat/IndicTrans2/blob/main/prepare_data_joint_training.sh
SRC_DIR=$1
TGT_DIR=$2
MODEL_FILE=$3
LANG=$4

mkdir -p $TGT_DIR

# check if the source language text requires transliteration
src_transliterate="true"
if [[ $LANG == *"Arab"* ]] || [[ $LANG == *"Olck"* ]] || \
    [[ $LANG == *"Mtei"* ]] || [[ $LANG == *"Latn"* ]]; then
    src_transliterate="false"
fi

echo "Normalizing punctuations"
bash normalize_punctuation.sh $LANG < $infname > $outfname._norm

