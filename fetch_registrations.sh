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

echo -e "\nDownloading TEMPEH registrations"
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=registrations/registrations.zip.001' -O "${download_dir}/registrations.zip.001" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=registrations/registrations.zip.002' -O "${download_dir}/registrations.zip.002" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=registrations/registrations.zip.003' -O "${download_dir}/registrations.zip.003" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=registrations/registrations.zip.004' -O "${download_dir}/registrations.zip.004" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=registrations/registrations.zip.005' -O "${download_dir}/registrations.zip.005" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=registrations/registrations.zip.006' -O "${download_dir}/registrations.zip.006" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=registrations/registrations.zip.007' -O "${download_dir}/registrations.zip.007" --no-check-certificate --continue

echo -e "\nUnzipping TEMPEH registrations"
7z x "${download_dir}/registrations.zip.001" -o"${output_dir}"

echo -e "\nDONE"
