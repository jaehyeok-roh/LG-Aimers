#!/bin/sh
# 로컬 30모델 학습이 끝나면 이어서 예측 상관을 잰다 (새벽 무인 실행).
# zip 이 생기고 **크기가 안정될 때까지** 기다린다 — 쓰는 중에 열면 깨진다.
cd /c/Users/nojh4/github/LG-Aimers || exit 1
Z=out/submit_local_cpu.zip

echo "[$(date +%H:%M)] 대기 시작 — $Z"
n=0
while [ ! -f "$Z" ]; do
  n=$((n + 1))
  [ $n -gt 240 ] && { echo "4시간 초과, 포기"; exit 1; }
  sleep 60
done

prev=0
while true; do
  cur=$(stat -c%s "$Z")
  if [ "$cur" = "$prev" ] && [ "$cur" -gt 1000000 ]; then break; fi
  prev=$cur
  sleep 30
done
echo "[$(date +%H:%M)] zip 완성 ($cur 바이트). 상관 측정 시작"

tail -25 out/train_local.log

PYTHONUTF8=1 python tools/pred_corr.py \
  out/submit_local_cpu.zip \
  /c/Users/nojh4/Downloads/submit_v9m.zip \
  "/g/내 드라이브/aimers_ablation/submit_s1lr.zip" \
  --rows=150000 2>&1

echo "[$(date +%H:%M)] 끝"
