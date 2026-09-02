# CAE 발표용 그림 다섯 장. tools/cae.py 가 저장한 out/ 산출물만 읽는다.
#
#   python tools/cae.py && python tools/cae_fig.py
#
# 재계산이 없으므로 그림과 발표 숫자가 어긋날 수 없다. 발표 자료의 모든 수치는
# out/cae_metrics.json 하나로 추적된다.
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 한글 축 라벨. 윈도우 기본 폰트가 없으면 조용히 네모가 되므로 확인하고 넘어간다.
for f in ('Malgun Gothic', 'AppleGothic', 'NanumGothic', 'DejaVu Sans'):
    if any(f == x.name for x in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = f
        print('폰트: %s' % f)
        break
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 160
plt.rcParams['savefig.bbox'] = 'tight'

# 그림과 그 근거가 되는 수치는 발표 자료 자산이므로 deck/ 아래에 둔다
# (out/ 은 .gitignore 대상이라 추적되지 않는다).
OUTD, FIGD = 'out', 'deck/fig'
os.makedirs(FIGD, exist_ok=True)
import shutil
shutil.copy(os.path.join(OUTD, 'cae_metrics.json'), 'deck/cae_metrics.json')
M = json.load(open(os.path.join(OUTD, 'cae_metrics.json'), encoding='utf-8'))
P24 = pd.read_csv(os.path.join(OUTD, 'cae_pitchers_2024.csv'), index_col=0)
P23 = pd.read_csv(os.path.join(OUTD, 'cae_pitchers_2023.csv'), index_col=0)

INK, ACC, WARN, MUTE = '#1b2430', '#2f6f8f', '#c2543a', '#9aa5b1'


def save(fig, name):
    p = os.path.join(FIGD, name)
    fig.savefig(p)
    plt.close(fig)
    print('  %s' % p)


# ── 1. CAE 분포 ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 3.4))
v = P24['cae'] * 100
ax.hist(v, bins=34, color=ACC, alpha=.85, edgecolor='white', linewidth=.6)
ax.axvline(0, color=INK, lw=1)
ax.set_xlabel('CAE (투구 100개당 추가 성공 수)')
ax.set_ylabel('투수 수')
# 문턱은 100 이다. P24.n.min() 은 실제 최솟값(104)이라 제목에 쓰면 오해를 준다.
ax.set_title('2024 CAE 분포 — 100구 이상 %d명' % len(P24), loc='left')
ax.text(.99, .93, '표준편차 %.1f\n상위10%% - 하위10%%  %.1f'
        % (v.std(), v.quantile(.9) - v.quantile(.1)),
        transform=ax.transAxes, ha='right', va='top', color=MUTE, fontsize=9)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
save(fig, '1_distribution.png')

# ── 2. 안정화 곡선: 몇 구부터 믿나 ───────────────────────────────────
st = {int(k): v for k, v in M['stability'].items()}
xs = sorted(st)
ys = [st[k]['rel'] for k in xs]
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(xs, ys, '-o', color=ACC, lw=2, ms=5)
ax.axhline(.8, color=WARN, ls='--', lw=1)
ax.text(xs[0], .81, '신뢰도 0.8', color=WARN, fontsize=9, va='bottom')
ax.set_xscale('log')
ax.set_xticks(xs)
ax.set_xticklabels([str(x) for x in xs])
ax.set_xlabel('그 투수의 최소 투구 수')
ax.set_ylabel('반쪽 분할 신뢰도')
ax.set_ylim(0, 1)
ax.set_title('언제부터 CAE 를 믿어도 되는가', loc='left')
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
save(fig, '2_stabilization.png')

# ── 3. 연도 간 이행: 표본이 적을수록 보정이 이긴다 ─────────────────
# 정직하게 교차점까지 보여준다. 300구를 넘으면 원시 성공률이 앞서는데, 그건
# 원시값이 '실력' 말고 **보직**까지 담고 있기 때문이다 (마무리는 내년에도
# 마무리다). CAE 는 그 성분을 일부러 걷어내므로 표본이 충분할 때 손해를 본다.
# 반대로 표본이 적으면 상황 보정이 그대로 이득이 된다.
T = {int(k.replace('구 이상', '')): v for k, v in M['transfer'].items()}
xs = sorted(T)
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6),
                         gridspec_kw={'width_ratios': [1.15, 1]})
ax = axes[0]
ax.plot(xs, [T[k]['cae'] for k in xs], '-o', color=ACC, lw=2, ms=5,
        label='CAE (상황 보정)')
ax.plot(xs, [T[k]['raw'] for k in xs], '-o', color=MUTE, lw=2, ms=5,
        label='원시 성공률')
ax.set_xscale('log')
ax.set_xticks(xs)
ax.set_xticklabels([str(x) for x in xs], fontsize=8)
ax.set_xlabel('양 시즌 최소 투구 수')
ax.set_ylabel('2023 -> 2024 상관')
ax.legend(frameon=False, fontsize=9, loc='lower right')
ax.set_title('표본이 적을수록 보정이 이긴다', loc='left', fontsize=11)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)

LO = 100
j2 = P23.query('n >= @LO')[['cae']].join(
    P24.query('n >= @LO')[['cae']], lsuffix='_23', rsuffix='_24', how='inner')
ax = axes[1]
x, y = j2['cae_23'] * 100, j2['cae_24'] * 100
ax.scatter(x, y, s=14, color=ACC, alpha=.65, edgecolor='none')
m = np.polyfit(x, y, 1)
xr = np.linspace(x.min(), x.max(), 2)
ax.plot(xr, np.polyval(m, xr), color=INK, lw=1.2)
ax.axhline(0, color=MUTE, lw=.6)
ax.axvline(0, color=MUTE, lw=.6)
ax.set_xlabel('2023 CAE')
ax.set_ylabel('2024 CAE')
ax.set_title('%d구 이상 %d명   r = %+.3f'
             % (LO, len(j2), np.corrcoef(x, y)[0, 1]), loc='left', fontsize=11)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
save(fig, '3_year_transfer.png')

# ── 4. 코치의 질문 넷 ────────────────────────────────────────────────
q = M['coach_questions']
order = [k for k in q if '최근' not in k] + [k for k in q if '최근' in k]
lab = {'주자 (클러치)': '"클러치에 강한가"\n주자 상황',
       '카운트 (불리)': '"불리해도 승부하나"\n볼카운트',
       '이닝 (경기 후반)': '"7회까지 맡기나"\n이닝',
       '점수차': '"점수차가 벌어지면"\n점수 상황',
       '최근 흐름 (시즌 내 전후반)': '"오늘 등판시키나"\n최근 흐름'}
vals = [q[k]['rel'] for k in order]
cols = [ACC if v > .15 else WARN for v in vals]
fig, ax = plt.subplots(figsize=(7.6, 3.6))
b = ax.bar(range(len(order)), vals, color=cols, width=.62)
ax.axhline(0, color=INK, lw=1)
ax.axhline(.15, color=MUTE, ls='--', lw=1)
ax.set_xticks(range(len(order)))
ax.set_xticklabels([lab.get(k, k) for k in order], fontsize=9)
ax.set_ylabel('축 내 CAE 신뢰도')
ax.set_ylim(min(0, min(vals)) - .05, max(vals) * 1.3)
for r, v in zip(b, vals):
    ax.text(r.get_x() + r.get_width() / 2, v + .02, '%.3f' % v, ha='center',
            fontsize=9, color=INK)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
save(fig, '4_coach_questions.png')

# ── 5. 실패 유형 신뢰도 ──────────────────────────────────────────────
fp = M['failure_profile']
ks = sorted(fp, key=lambda k: -fp[k]['rel'])
fig, ax = plt.subplots(figsize=(7, 3.3))
cols = [WARN if k == '반대방향' else (INK if k == '성공' else MUTE) for k in ks]
b = ax.barh(range(len(ks))[::-1], [fp[k]['rel'] for k in ks], color=cols,
            height=.6)
ax.set_yticks(range(len(ks))[::-1])
ax.set_yticklabels(ks)
ax.set_xlabel('투수 수준 반쪽 분할 신뢰도')
ax.set_xlim(0, 1)
for r, k in zip(b, ks):
    ax.text(fp[k]['rel'] + .012, r.get_y() + r.get_height() / 2,
            '%.3f' % fp[k]['rel'], va='center', fontsize=9, color=INK)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
save(fig, '5_failure_profile.png')

print('그림 5장 저장 완료')
