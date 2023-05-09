#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

if [ "$#" -eq 0 ]; then
	echo -e "\nBefore you continue, you must register at https://tempeh.is.tue.mpg.de/ and agree to the TEMPEH license terms."
	read -p "Username (TEMPEH):" username
	read -p "Password (TEMPEH):" password
    download_dir='./data/downloads'
    output_dir='./data/test_data'
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


echo -e "\nDownloading TEMPEH test images"
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=test_data/test_images_4.zip.001' -O "${download_dir}/test_images_4.zip.001" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=test_data/test_images_4.zip.002' -O "${download_dir}/test_images_4.zip.002" --no-check-certificate --continue

sleep 10s

echo -e "\nDownloading TEMPEH test camera calibrations"
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=test_data/test_calibrations.zip.001' -O "${download_dir}/test_calibrations.zip.001" --no-check-certificate --continue

sleep 10s

echo -e "\nDownloading TEMPEH test list"
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=test_data/fiveteen_subj__all_seq_frames_per_seq_20_test.json' -O "${output_dir}/fiveteen_subj__all_seq_frames_per_seq_20_test.json" --no-check-certificate --continue

sleep 10s

echo -e "\nUnzipping TEMPEH test images"
7z x "${download_dir}/test_images_4.zip.001" -o"${output_dir}"

sleep 10s

echo -e "\nUnzipping TEMPEH test camera calibrations"
unzip "${download_dir}/test_calibrations.zip.001" -d "${output_dir}"

sleep 10s

echo -e "\nDONE"
