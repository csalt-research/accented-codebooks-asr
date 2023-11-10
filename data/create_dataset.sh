#!/usr/bin/env bash

# Folder to host all created csv splits
helper_datafolder='helper_data'
datafolder='accent_splits'
helper_tarfile='helper.tar.gz'
tarfile='dataset.tar.gz'

# Our script needs some files from the original commonvoice dataset. These files already exist in the tar `data.tar.gz`.
if [ ! -d ${datafolder} ]; then
  echo "Unzipping required files....."
  tar -xvzf $helper_tarfile
  echo ""
fi

# mkdir -p $datafolder || exit 1

# Run script to generate the `MCV_ACCENT` train, dev and test splits
python3 scripts/create_dataset.py --input-dir ${helper_datafolder}/  --output-dir ${datafolder}/csvs/

# # Run script to subsample from the `MCV_ACCENT` dataset to generate `MCV_ACCENT_100` data splits.
python3 scripts/subsample_dataset.py --input-dir ${datafolder}/csvs

# Create a tar file containing all the csvs
echo ""
echo "Zipping files....."
tar -cvzf ${tarfile} ${datafolder}

