#!/bin/bash
# 캐글 동시 CPU 세션 상한은 5개다. 자리가 나면 대기 커널을 올린다.
cd /c/Users/nojh4/github/LG-Aimers
PENDING="$@"
for k in $PENDING; do
  until (cd ".kernels/$k" && PYTHONUTF8=1 kaggle kernels push -p . 2>&1 | grep -q "successfully pushed"); do
    sleep 300
  done
  echo "[$(date +%H:%M)] $k 올림"
done
echo QUEUE_DONE
