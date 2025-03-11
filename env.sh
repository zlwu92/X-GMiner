#!/bin/bash

fullpath=$(readlink --canonicalize --no-newline $BASH_SOURCE)
cur_dir=$(cd `dirname ${fullpath}`; pwd)

export XGMINER_ROOT="${cur_dir}/xgminer/"

export PATH="${XGMINER_ROOT}:${PATH}"
echo "ROOT_PATH set to: $XGMINER_ROOT"
