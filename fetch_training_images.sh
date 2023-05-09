#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

if [ "$#" -eq 0 ]; then
	echo -e "\nBefore you continue, you must register at https://tempeh.is.tue.mpg.de/ and agree to the TEMPEH license terms."
	read -p "Username (TEMPEH):" username
	read -p "Password (TEMPEH):" password
	download_dir='./data/downloads'
	output_dir='./data/training_data'
else
	username="$1"
	password="$2"
	download_dir="$3"
	output_dir="$4"
fi

username=$(urle "$username")
password=$(urle "$password")

mkdir -p "$download_dir"
mkdir -p "$output_dir"

echo -e "\nDownloading TEMPEH training images"
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=training_data/images/training_images_4.zip.001' -O "${download_dir}/training_images_4.zip.001" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=training_data/images/training_images_4.zip.002' -O "${download_dir}/training_images_4.zip.002" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=training_data/images/training_images_4.zip.003' -O "${download_dir}/training_images_4.zip.003" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=training_data/images/training_images_4.zip.004' -O "${download_dir}/training_images_4.zip.004" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=training_data/images/training_images_4.zip.005' -O "${download_dir}/training_images_4.zip.005" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=training_data/images/training_images_4.zip.006' -O "${download_dir}/training_images_4.zip.006" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=training_data/images/training_images_4.zip.007' -O "${download_dir}/training_images_4.zip.007" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=training_data/images/training_images_4.zip.008' -O "${download_dir}/training_images_4.zip.008" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=training_data/images/training_images_4.zip.009' -O "${download_dir}/training_images_4.zip.009" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=training_data/images/training_images_4.zip.010' -O "${download_dir}/training_images_4.zip.010" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=training_data/images/training_images_4.zip.011' -O "${download_dir}/training_images_4.zip.011" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=training_data/images/training_images_4.zip.012' -O "${download_dir}/training_images_4.zip.012" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=training_data/images/training_images_4.zip.013' -O "${download_dir}/training_images_4.zip.013" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=training_data/images/training_images_4.zip.014' -O "${download_dir}/training_images_4.zip.014" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=training_data/images/training_images_4.zip.015' -O "${download_dir}/training_images_4.zip.015" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=training_data/images/training_images_4.zip.016' -O "${download_dir}/training_images_4.zip.016" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=training_data/images/training_images_4.zip.017' -O "${download_dir}/training_images_4.zip.017" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=training_data/images/training_images_4.zip.018' -O "${download_dir}/training_images_4.zip.018" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=training_data/images/training_images_4.zip.019' -O "${download_dir}/training_images_4.zip.019" --no-check-certificate --continue


echo -e "\nDownloading TEMPEH training camera calibrations"
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=training_data/calibrations/training_calibrations.zip.001' -O "${download_dir}/training_calibrations.zip.001" --no-check-certificate --continue


echo -e "\nDownloading TEMPEH training and validation lists"
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=training_data/seventy_subj__all_seq_frames_per_seq_40_head_rot_120_train.json' -O "${output_dir}/seventy_subj__all_seq_frames_per_seq_40_head_rot_120_train.json" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=training_data/eight_subj__all_seq_frames_per_seq_5_val.json' -O "${output_dir}/eight_subj__all_seq_frames_per_seq_5_val.json" --no-check-certificate --continue


echo -e "\nUnzipping TEMPEH training images"
7z x "${download_dir}/training_images_4.zip.001" -o"${output_dir}"

echo -e "\nUnzipping TEMPEH training camera calibrations"
unzip "${download_dir}/training_calibrations.zip.001" -d "${output_dir}"

echo -e "\nDONE"
