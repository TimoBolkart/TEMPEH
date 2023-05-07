#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

echo -e "\nBefore you continue, you must register at https://tempeh.is.tue.mpg.de/ and agree to the TEMPEH license terms."
read -p "Username (TEMPEH):" username
read -p "Password (TEMPEH):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p './runs/coarse'
mkdir -p './runs/refinement'

echo -e "\nDownloading TEMPEH coarse model"
mkdir -p './data/downloads'
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=models/coarse__TEMPEH_final.zip' -O './data/downloads/coarse__TEMPEH_final.zip' --no-check-certificate --continue

echo -e "\nDownloading TEMPEH refinement model"
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=tempeh&resume=1&sfile=models/refinement_TEMPEH_final.zip' -O './data/downloads/refinement_TEMPEH_final.zip' --no-check-certificate --continue

echo -e "\nUnzipping TEMPEH coarse model"
unzip './data/downloads/coarse__TEMPEH_final.zip' -d './runs/coarse'

echo -e "\nUnzipping TEMPEH refinement model"
unzip './data/downloads/refinement_TEMPEH_final.zip' -d './runs/refinement'

echo -e "\nDONE"
