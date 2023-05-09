#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

echo -e "\nBefore you continue, you must register at https://tempeh.is.tue.mpg.de/ and agree to the TEMPEH license terms."
read -p "Username (TEMPEH):" username
read -p "Password (TEMPEH):" password

download_dir='./data/downloads'
output_dir='./data/training_data'

./fetch_training_images.sh "$username" "$password" "$download_dir" "$output_dir"
./fetch_training_scans.sh "$username" "$password" "$download_dir" "$output_dir"
./fetch_registrations.sh "$username" "$password" "$download_dir" "$output_dir"
