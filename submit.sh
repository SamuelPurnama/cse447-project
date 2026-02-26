#!/usr/bin/env bash
set -x
set -e

rm -rf submit submit.zip
mkdir -p submit

# submit team.txt
printf "JungHo Park, jpark132\nYining Zhong, yininz6\nSamuel Purnama, samjp53" > submit/team.txt

# prepare datasets (download + split into English/non-English under work/)
python src/prepare_datasets.py --work_dir work --dataset opensubtitles --download_if_missing

# download OpenSubtitles with English/non-English splits (writes to work/data/)
python src/download_opensubtitles_with_splits.py --output_dir work/data --langs hi,ja,es,ko --all

# train model
python src/myprogram.py train --work_dir work

# make predictions on example data submit it in pred.txt
python src/myprogram.py test --work_dir work --test_data example/input.txt --test_output submit/pred.txt

# submit docker file
cp Dockerfile submit/Dockerfile

# submit requirements.txt
cp requirements.txt submit/requirements.txt

# submit source code
cp -r src submit/src

# submit checkpoints
cp -r work submit/work

# make zip file
zip -r submit.zip submit
