#!/bin/bash

set -e
if [ "$#" -ne 2 ]; then
    echo "Usage: create-github-issue <title> <body>"
    exit 1
fi
title=$1
body=$2
curl -u $GITHUB_TOKEN:x-oauth-basic -i -H "Content-Type: application/json" -X POST --data '{"title":"'$title'", "body":"'$body'"}' https://api.github.com/repos/pytroll/pygac-fdr/issues