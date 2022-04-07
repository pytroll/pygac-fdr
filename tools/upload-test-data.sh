#!/bin/bash
# Upload test data to DWD server.

if [ "$#" -ne 1 ]; then
    echo "Usage: upload-test-data.sh <test data dir>"
    exit 1
fi

source_dir=$1
target_dir=perm/pygac-fdr/test_data
upload='/cmsaf/nfshome/routcm/common/bin/put_public_cmsaf'

# Delete existing data
$upload -D perm/pygac-fdr/test_data/
sleep 30

# Upload new data
set -e

$upload -p $target_dir/input/corrupt $source_dir/input/corrupt/NSS*
sleep 30
$upload -p $target_dir/input/normal $source_dir/input/normal/*.gz
sleep 30
$upload -p $target_dir/tle $source_dir/tle/*.txt
sleep 10
$upload -p $target_dir/output_ref/normal $source_dir/output_ref/normal/*.nc
sleep 30
$upload -p $target_dir/output_ref/corrupt $source_dir/output_ref/corrupt/*.nc
