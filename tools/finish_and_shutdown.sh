#!/bin/bash
# eda29 가 끝나면 결과를 리포에 저장 -> 커밋/푸시 -> 컴퓨터 종료.
#
# 왜 저장부터 하나: eda29 출력은 지금 임시 폴더(AppData\Local\Temp)에만 있다.
# 재부팅으로 지워질 수 있으므로 리포에 넣고 깃허브로 밀어올린 뒤에 끈다.
cd /c/Users/nojh4/github/LG-Aimers
OUT="C:/Users/nojh4/AppData/Local/Temp/claude/C--Users-nojh4-github-LG-Aimers/c7db979e-c014-4774-9c63-b77e9a84a2c4/tasks/bsi8u6rep.output"
DEST="out/eda29_result.txt"
DEADLINE=$((SECONDS+14400))     # 4시간 안전장치 — 무슨 일이 있어도 그땐 끈다

echo "[$(date +%H:%M)] eda29 종료 대기 시작"
while [ $SECONDS -lt $DEADLINE ]; do
  # eda29 는 마지막에 '읽는 법' 을 찍는다. 파이프(tail) 때문에 종료 시점에 한꺼번에 쓰인다.
  if [ -s "$OUT" ] && grep -q "읽는 법" "$OUT" 2>/dev/null; then
    echo "[$(date +%H:%M)] eda29 정상 종료 감지"; break
  fi
  # 파이썬이 죽었는데 결과가 없으면(크래시) 더 기다릴 이유가 없다
  if ! tasklist //FI "IMAGENAME eq python.exe" 2>/dev/null | grep -qi python.exe; then
    echo "[$(date +%H:%M)] 파이썬 프로세스 없음 — 크래시했거나 이미 끝났다"; break
  fi
  sleep 60
done

mkdir -p out
{ echo "# eda29 — 당해 시즌 폼 축 잔량 측정"
  echo "# 저장 $(date '+%Y-%m-%d %H:%M')"
  echo
  cat "$OUT" 2>/dev/null || echo "(출력 없음 — 재실행 필요)"
} > "$DEST"
echo "[$(date +%H:%M)] $DEST 저장 ($(wc -l < "$DEST") 줄)"

git add -f "$DEST" 2>/dev/null
git commit -q -m "eda29 결과: 당해 시즌 폼 축 잔량 (자동 저장)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" 2>/dev/null \
  && git push -q origin main 2>/dev/null && echo "[$(date +%H:%M)] 푸시 완료" \
  || echo "[$(date +%H:%M)] 커밋/푸시 생략 (변경 없음이거나 실패)"

echo "[$(date +%H:%M)] 3분 뒤 종료 예약 — 취소하려면:  shutdown /a"
shutdown //s //t 180 //c "eda29 finished - auto shutdown. Cancel: shutdown /a"
