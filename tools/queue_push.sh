#!/bin/bash
# 캐글 동시 CPU 세션 상한은 5개다. 자리가 나면 대기 커널을 올린다.
#
# ⚠️ **이미 RUNNING/COMPLETE 인 커널은 절대 다시 push 하지 않는다.**
#    push 는 새 버전을 만들면서 **돌던 커널을 죽이고 처음부터 다시 시작시킨다.**
#    2026-08-23 에 우선순위를 바꾸려고 대기열을 네 번 재시작했는데, 그때마다
#    맨 앞의 wbcc(4.5시간짜리)가 재시작돼서 7시간이 지나도 안 끝났다.
#    캐글에 버전이 6개 쌓인 것을 사용자가 발견해서 알았다.
cd /c/Users/nojh4/github/LG-Aimers

st() {
  PYTHONUTF8=1 kaggle kernels status "your-kaggle-id/aimers-$1" 2>/dev/null \
    | tr -d '\r' | grep -o 'KernelWorkerStatus\.[A-Z]*' | cut -d. -f2
}

for k in "$@"; do
  s=$(st "$k")
  if [ "$s" = "RUNNING" ] || [ "$s" = "COMPLETE" ]; then
    echo "[$(date +%H:%M)] $k 는 이미 $s — 건너뜀"
    continue
  fi
  while true; do
    s=$(st "$k")
    if [ "$s" = "RUNNING" ] || [ "$s" = "COMPLETE" ]; then
      echo "[$(date +%H:%M)] $k 가 $s 로 바뀜 — 건너뜀"; break
    fi
    if (cd ".kernels/$k" && PYTHONUTF8=1 kaggle kernels push -p . 2>&1 \
        | grep -q "successfully pushed"); then
      echo "[$(date +%H:%M)] $k 올림"; break
    fi
    sleep 300
  done
done
echo QUEUE_DONE
