TARGET=ttbar2024pu

FILE_LIST=($(ls /home/joshin/workspace-gate/store-hdfs/DeepMuonReco/CMSSW_14_0_21_patch1/${TARGET}/*TFile-*.root | sort -V))
echo "FILE_LIST count: ${#FILE_LIST[@]}"

OUT_DIR=/home/joshin/workspace-gate/DeepMuonReco/data/${TARGET}1M
mkdir -p $OUT_DIR

TRAIN_FILES=$(printf "%s\n" "${FILE_LIST[@]:0:640}")
hadd -f $OUT_DIR/train.root $TRAIN_FILES

VAL_FILES=$(printf "%s\n" "${FILE_LIST[@]:640:160}")
hadd -f $OUT_DIR/val.root $VAL_FILES

TEST_FILES=$(printf "%s\n" "${FILE_LIST[@]:800:200}")
hadd -f $OUT_DIR/test.root $TEST_FILES

SANITY_CHECK=$(printf "%s\n" "${FILE_LIST[@]:0:1}")
hadd -f $OUT_DIR/sanity-check.root $SANITY_CHECK

#TRAIN_FILES=$(printf "%s\n" "${FILE_LIST[@]:0:60}")
#hadd -f $OUT_DIR/train.root $TRAIN_FILES
#
#VAL_FILES=$(printf "%s\n" "${FILE_LIST[@]:60:20}")
#hadd -f $OUT_DIR/val.root $VAL_FILES
#
#TEST_FILES=$(printf "%s\n" "${FILE_LIST[@]:80:20}")
#hadd -f $OUT_DIR/test.root $TEST_FILES
#
#SANITY_CHECK=$(printf "%s\n" "${FILE_LIST[@]:0:1}")
#hadd -f $OUT_DIR/sanity-check.root $SANITY_CHECK
