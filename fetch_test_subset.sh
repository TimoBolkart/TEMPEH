#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

echo -e "\nBefore you continue, you must register at https://tempeh.is.tue.mpg.de/ and agree to the TEMPEH license terms."
read -p "Username (TEMPEH):" username
read -p "Password (TEMPEH):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p './data/test_data_subset'

echo -e "\nDownloading TEMPEH test subset images"
mkdir -p './data/downloads'
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=test_subset_data/test_subset_images_4.zip.zip' -O './data/downloads/test_subset_images_4.zip' --no-check-certificate --continue

echo -e "\nDownloading TEMPEH test subset camera calibrations"
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=test_subset_data/test_subset_calibrations.zip' -O './data/downloads/test_subset_calibrations.zip' --no-check-certificate --continue

echo -e "\nDownloading TEMPEH test subset list"
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=test_subset_data/paper_test_frames.json' -O './data/test_data_subset/paper_test_frames.json' --no-check-certificate --continue

echo -e "\nUnzipping TEMPEH test subset images..."
unzip './data/downloads/test_subset_images_4.zip' -d './data/test_data_subset/test_subset_images_4'

echo -e "\nUnzipping TEMPEH test subset camera calibrations"
unzip './data/downloads/test_subset_calibrations.zip' -d './data/test_data_subset/test_subset_calibrations'

echo -e "\nDONE"
