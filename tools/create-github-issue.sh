#!/bin/bash
# Create a pytroll/pygac-fdr issue with the given title and body. This has been moved to a
# separate script because commands containing colons invalidate travis.yaml .
set -e
if [ "$#" -ne 2 ]; then
    echo "Usage: create-github-issue <title> <body>"
    exit 1
fi
title="$1"
body="$2"
curl \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: token ${GITHUB_TOKEN}" \
  -X POST \
  --data "{\"title\":\"$title\", \"body\":\"$body\"}" \
  "https://api.github.com/repos/pytroll/pygac-fdr/issues"


