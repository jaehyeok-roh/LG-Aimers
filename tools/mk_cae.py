# CAE 용 홀드아웃 예측을 생성하는 커널을 만든다 (본 학습 없음).
#
#   python tools/mk_cae.py
#
# CAE(Command Above Expected) = mean(실제 성공 - 모델 기대 성공) 이다.
# **기대값이 정직해야 지표가 의미가 있다.**
#
# ⚠️ 배포 모델(`aimers_v10wtor.ipynb` 의 셀 18)은 2024 를 학습에 포함하므로
#    그대로 쓰면 in-sample 암기로 CAE 분산이 부풀고 신뢰도가 가짜로 높아진다.
#    이 프로젝트에서 이미 그 함정으로 λ*=1.687 이라는 가짜 +270점을 본 적이 있다
#    (claude.md eda61 — 암기 0.4% 면 그 크기가 전부 설명된다).
#
# 그래서 **오프셋 셀(셀 20)의 구조를 그대로 재사용한다** — `~Y-1` 3-fold 학습 ->
# `Y` 예측 -> 폴드별 isotonic -> 3폴드 평균. 이건 배포 시 재중심화 상수를 재는
# 데 쓰던 검증된 경로이고, 검증 시즌 라벨을 안 본 보조 피처(AUX_HO)까지 갈아끼운다.
#
# 두 시즌(2023·2024)을 뽑는다. 2023 은 **연도 간 이행 검증**에 쓴다 —
# "2023 CAE 가 2024 CAE 를 예측하는가" 가 선수 지표의 진짜 시험이다.
#
# 본 학습 30모델(약 45분)은 잘라낸다. 필요한 건 예측뿐이다.
import ast
import json
import os
import sys

BASE = os.environ.get('CAE_BASE', 'experiments/v10w/aimers_v10wtor.ipynb')
OUT = os.environ.get('CAE_OUT', 'experiments/v10w/aimers_cae.ipynb')
SEASONS = [int(x) for x in os.environ.get('CAE_SEASONS', '2023,2024').split(',')]

nb = json.load(open(BASE, encoding='utf-8'))
cells = nb['cells']
code = [i for i, c in enumerate(cells) if c['cell_type'] == 'code']
src = lambda i: ''.join(cells[i]['source'])


def find(a):
    h = [i for i in code if a in src(i)]
    if len(h) != 1:
        sys.exit('앵커 %d곳: %s' % (len(h), a[:90]))
    return h[0]


# ── 1) 본 학습 루프를 잘라낸다. 앞부분(임포트 · X,y · 헬퍼)은 셀 20 이 쓰므로 남긴다
i_tr = find('=== 최종 학습:')
s = src(i_tr)
CUT = 'seed_oof_raw = {s: np.zeros(len(X)) for s in SEEDS}'
if CUT not in s:
    sys.exit('학습 루프 시작 앵커를 못 찾았다')
head = s[:s.index(CUT)]
cells[i_tr]['source'] = (head + '\nprint("본 학습 생략 (CAE 용 예측만 만든다)")\n'
                         ).splitlines(keepends=True)
print('셀 %d: 본 학습 루프 제거 (%d -> %d자)' % (i_tr, len(s), len(head)))

# ── 2) 오프셋 셀을 시즌 루프로 감싸고 예측을 저장한다
i_off = find('solve_logit_offset')
s = src(i_off)
A = 'RECENTER_OFFSET = 0.0\nif RECENTER:'
if A not in s:
    sys.exit('오프셋 셀 앵커를 못 찾았다')
# 앞부분에 solve_logit_offset 정의가 있다. 본문이 그걸 호출하므로 반드시 살린다
# (nbcheck 가 '정의 전 사용' 으로 잡아준 자리다).
HEAD = s[:s.index(A)]
body = s[s.index(A) + len(A):]
# 저장 블록 이후(train_constants 갱신)는 CAE 커널에 불필요하므로 자른다
END = '# train_constants.json 갱신'
if END in body:
    body = body[:body.index(END)]

NEW = '''# ================= CAE 용 홀드아웃 예측 (mk_cae) =================
# 배포 노트북의 오프셋 셀 구조를 그대로 쓴다: ~Y-1 로 3-fold 학습 -> Y 예측 ->
# 폴드별 isotonic -> 평균. **Y 는 학습에 들어가지 않는다** = CAE 의 기대값이 정직하다.
import numpy as _np

CAE_SEASONS = %r
_cae_out = {}
for HOLDOUT_SEASON in CAE_SEASONS:
    print("\\n" + "=" * 60)
    print("CAE 예측: ~%%d 학습 -> %%d 예측" %% (HOLDOUT_SEASON - 1, HOLDOUT_SEASON))
%s
    _cae_out[str(HOLDOUT_SEASON)] = {
        "row_id": df_processed.loc[_va_m, "row_id"].to_numpy().astype(str),
        "pred": _ph.astype("float64"),
        "y": _yv.to_numpy().astype("float64"),
    }
    # 학습 시즌이 정말 Y 를 안 봤는지 확인한다 (CAE 전체가 여기 달려 있다)
    _tr_seasons = sorted(df_processed.loc[_tr_m, "season"].unique().tolist())
    assert HOLDOUT_SEASON not in _tr_seasons, (HOLDOUT_SEASON, _tr_seasons)
    print("  학습 시즌 %%s  (검증 %%d 미포함 확인)" %% (_tr_seasons, HOLDOUT_SEASON))
    print("  검증 %%s행 | 평균예측 %%.4f | 실제 %%.4f"
          %% (format(len(_ph), ","), _ph.mean(), _yv.mean()))

_np.savez("cae_pred.npz",
          **{"%%s_%%s" %% (k, f): v for k, d in _cae_out.items()
             for f, v in d.items()})
print("\\ncae_pred.npz 저장:", sorted(_cae_out))
''' % (SEASONS, body.rstrip())

# 오프셋 셀을 통째로 교체
cells[i_off]['source'] = (HEAD + NEW).splitlines(keepends=True)
print('셀 %d: 오프셋 셀 -> 시즌 %s 예측 저장으로 교체' % (i_off, SEASONS))

# ── 3) script.py · zip 조립 · 검증 샌드박스 셀은 제거 (제출물이 아니다)
drop = []
for a in ('SCRIPT_TEMPLATE', 'cb_files = sorted', 'SANDBOX = "validation_sandbox"'):
    h = [i for i in code if a in src(i)]
    drop += h
for i in sorted(set(drop), reverse=True):
    cells[i]['source'] = ['print("CAE 커널에서는 생략")\n']
print('셀 %s: 제출물 조립 단계 비활성화' % sorted(set(drop)))

for i in code:
    ast.parse(src(i))
json.dump(nb, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('%s 생성 -- %d셀 문법 OK' % (OUT, len(code)))
