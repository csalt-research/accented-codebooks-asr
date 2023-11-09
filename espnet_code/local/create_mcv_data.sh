#!/usr/bin/env bash
# Given path to a csv file, the sh file generated all files required by espnet.
# Note: The csv file should have `client_id`, `path`, `sentence`, `accent`, `duration` in the same order

if [ $# -ne 3 ]; then
  echo "Usage: $0 <path-to-csv> <path-to-clips-directory> <destination-directory>"
  exit 1
fi

datafile=$1
clips_folder=$2
destination=$3

mkdir -p $destination || exit 1

if [ ! -f "$datafile" ]; then
  echo "$0: no such directory $datafile"
  exit 1
fi

if [ ! -d "$clips_folder" ]; then
  echo "$0: no such directory $clips_folder"
  exit 1
fi

head $datafile

# Generate `text` file
cat $datafile | awk -F'\t' 'BEGIN{FPAT = "([^\t]+)|(\"[^\"]+\")"} NR>1 { sub(".mp3","",$2);gsub("-"," ",$3);gsub("[^a-zA-Z0-9 ]","",$3); print $1"-"$2" "toupper($3)}' | sort > $destination/text

# Generate `wav.scp` file
cat $datafile | awk -F'\t' -v clips_folder="$clips_folder" 'BEGIN{FPAT = "([^\t]+)|(\"[^\"]+\")"} NR>1 { sub(".mp3","",$2); print $1"-"$2" ffmpeg -i " clips_folder "/" $2".mp3  -f wav -ar 16000 -ab 16 -ac 1 - | sox -t wav - -t wav - | "}' | sort > $destination/wav.scp

# Generate `utt2spk` file
cat $datafile | awk -F',' 'BEGIN{FPAT = "([^\t]+)|(\"[^\"]+\")"} NR>1 { sub(".mp3","",$2); print $1"-"$2" "$1}' | sort > $destination/utt2spk

# Generate `spk2utt` file
utils/utt2spk_to_spk2utt.pl $destination/utt2spk > $destination/spk2utt

# cat $datafile | awk -F',' 'BEGIN{FPAT = "([^\t]+)|(\"[^\"]+\")"} NR>1 { sub(".mp3","",$2); print $1"-"$2" "$5}' | sort > $destination/utt2dur

utils/validate_data_dir.sh --no-feats $destination || exit 1
