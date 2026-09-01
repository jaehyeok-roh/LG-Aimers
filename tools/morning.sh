#!/bin/bash
# 아침에 이거 하나만 돌리면 된다.
#   1) 끝난 캐글 커널 산출물을 전부 받는다
#   2) 스크리너 결과를 뽑아 보여준다
#   3) CPU 30모델이 다 모였으면 조립 + 독립성 검증까지 한다
cd /c/Users/nojh4/github/LG-Aimers

# 밤새 슬롯이 없어 못 올라간 커널이 있으면 먼저 올린다
echo "===== 미실행 커널 push ====="
for k in ktsk; do
  st=$(PYTHONUTF8=1 kaggle kernels status your-kaggle-id/aimers-$k 2>/dev/null | tr -d '
' | grep -o 'KernelWorkerStatus\.[A-Z]*' | cut -d. -f2)
  if [ "$st" != "RUNNING" ] && [ ! -f "kaggle_output/$k/aimers-$k.log" ]; then
    (cd ".kernels/$k" && PYTHONUTF8=1 kaggle kernels push -p . 2>&1 | tail -1)
  fi
done

echo "===== 커널 상태 ====="
for k in cpua cpub cpuc kgt1 kgt2 ktsk k23 kbat; do
  s=$(PYTHONUTF8=1 kaggle kernels status your-kaggle-id/aimers-$k 2>/dev/null \
      | tr -d '\r' | grep -o 'KernelWorkerStatus\.[A-Z]*' | cut -d. -f2)
  printf '  %-6s %s\n' "$k" "${s:-없음}"
  # 디렉터리 존재만 보면 안 된다 — 종료로 중단된 부분 다운로드가 남아 있을 수 있다.
  # 로그 파일이 실제로 있는지로 판정한다.
  if [ "$s" = "COMPLETE" ] && [ ! -f "kaggle_output/$k/aimers-$k.log" ]; then
    rm -rf "kaggle_output/$k"
    mkdir -p "kaggle_output/$k"
    PYTHONUTF8=1 kaggle kernels output "your-kaggle-id/aimers-$k" -p "kaggle_output/$k" >/dev/null 2>&1
    echo "         ↳ 받음"
  fi
done

echo; echo "===== 스크리너 결과 ====="
PYTHONUTF8=1 python - <<'PY'
import glob, json, os
for f in sorted(glob.glob('kaggle_output/*/aimers-*.log')):
    tag = os.path.basename(os.path.dirname(f))
    try:
        t = ''.join(d.get('data','') for d in json.load(open(f,encoding='utf-8'))
                    if isinstance(d,dict))
    except Exception:
        continue
    hits = [l.strip() for l in t.splitlines()
            if '점  (' in l or l.strip().startswith('"')]
    if hits:
        print(f'  [{tag}]')
        for h in hits[:8]:
            print(f'    {h}')
PY
echo "  (기준: base 813.4 / wseason5 885.6 — wseason5 대비 +50 넘어야 의미)"

echo; echo "===== CPU 30모델 조립 ====="
PYTHONUTF8=1 python tools/assemble_cpu.py && {
  echo; echo "===== 독립성 검증 (규정) ====="
  PYTHONUTF8=1 python tools/audit_independence.py out/submit_cpu30.zip 800 3
}

echo; echo "===== 다음 ====="
echo "  통과했으면 out/submit_cpu30.zip 제출 (기대 ~1064)"
echo "  eda30 실행(eda29 는 역산 누수로 무효):  PYTHONUTF8=1 python tools/eda30.py 1000"
echo "  대기 커널 push: bash tools/queue_push.sh k23 kbat"
