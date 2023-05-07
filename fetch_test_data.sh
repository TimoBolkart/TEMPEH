#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

echo -e "\nBefore you continue, you must register at https://tempeh.is.tue.mpg.de/ and agree to the TEMPEH license terms."
read -p "Username (TEMPEH):" username
read -p "Password (TEMPEH):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p './data/test_data'

echo -e "\nDownloading TEMPEH test images"
mkdir -p './data/downloads'
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=test_data/test_images_4.zip.001' -O './data/downloads/test_images_4.zip.001' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=test_data/test_images_4.zip.002' -O './data/downloads/test_images_4.zip.002' --no-check-certificate --continue

echo -e "\nDownloading TEMPEH test camera calibrations"
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=test_data/test_calibrations.zip.001' -O './data/downloads/test_calibrations.zip.001' --no-check-certificate --continue

echo -e "\nDownloading TEMPEH test list"
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=test_data/fiveteen_subj__all_seq_frames_per_seq_20_test.json' -O './data/test_data/fiveteen_subj__all_seq_frames_per_seq_20_test.json' --no-check-certificate --continue

echo -e "\nUnzipping TEMPEH test images"
7z x './data/downloads/test_images_4.zip.001' -o'./data/test_data/test_images_4'

echo -e "\nUnzipping TEMPEH test camera calibrations"
unzip './data/downloads/test_calibrations.zip.001' -d './data/test_data/test_calibrations'

echo -e "\nDONE"
