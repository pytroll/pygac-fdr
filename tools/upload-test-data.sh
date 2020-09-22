#!/bin/bash
# Upload test data to DWD server.

upload='/cmsaf/nfshome/routcm/common/bin/put_public_cmsaf'

# Delete existing data
$upload -D perm/pygac-fdr/test_data/
sleep 30

# Upload new data
set -e
$upload -p perm/pygac-fdr/test_data/input/corrupt input/corrupt/NSS*
sleep 30
$upload -p perm/pygac-fdr/test_data/input/normal input/normal/*.gz
sleep 30
$upload -p perm/pygac-fdr/test_data/tle tle/*.txt
sleep 10
$upload -p perm/pygac-fdr/test_data/output_ref/normal output_ref/normal/*.nc
sleep 30
$upload -p perm/pygac-fdr/test_data/output_ref/corrupt output_ref/corrupt/*.nc
