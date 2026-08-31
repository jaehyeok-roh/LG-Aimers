# 리더보드에서 **100번째 개인** 컷을 재고 우리 위치를 찾는다. 학습 0.
#
#   python tools/leaderboard.py            (내려받아서 파싱)
#   python tools/leaderboard.py saved.html (이미 받아둔 파일)
#
# ⚠️ 함정 둘. 둘 다 실제로 밟았다.
#
# 1) **점수가 두 형태로 섞여 있다.** Nuxt SSR 페이로드가 일부 항목은 압축 리터럴
#    `{...,team_name:"x",score:1141.93,...}` 로, 일부는 최소화 대입
#    `fu.team_name="x";fu.score=1153.60;` 로 쓴다. `score:` 만 잡으면 873/973 개만
#    걸리고, 하필 우리 항목이 대입 형태라 **"우리가 1위" 라는 결론이 나온다.**
#    두 형태를 모두 잡아야 한다.
#
# 2) **컷은 팀이 아니라 개인 100명 기준이다.** 상위 팀은 평균 2.8명이라
#    팀 순위와 개인 순위가 3배 가까이 벌어진다 (우리는 팀 71위 = 개인 210번째).
#    `team_info` 안의 `user_id` 개수를 누적해야 한다.
import re
import subprocess
import sys
import tempfile
import os

URL = 'https://dacon.io/competitions/official/236743/leaderboard'
OURS = os.environ.get('LB_TEAM', '앙대ai')

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = os.path.join(tempfile.gettempdir(), 'lb.html')
    subprocess.run(['curl', '-s', URL, '-H', 'User-Agent: Mozilla/5.0',
                    '-o', path], check=True)
h = open(path, encoding='utf-8', errors='replace').read()

rows = []
for p in re.split(r'team_name[:=]', h)[1:]:
    m = re.search(r'score[:=](\d+\.\d+)', p[:2000])
    if not m:
        continue
    nm = re.match(r'\s*"([^"]*)"', p)
    seg = p[:p.find('team_info') + 4000] if 'team_info' in p else p[:2000]
    ti = re.search(r'team_info[:=]\[(.*?)\](?=;|,\w+:|\})', seg, re.S)
    n = len(re.findall(r'user_id', ti.group(1))) if ti else 1
    rows.append((float(m.group(1)), nm.group(1) if nm else '?', max(n, 1)))
rows.sort(key=lambda r: -r[0])
print('%d팀 / %d명 | 최고 %.4f' % (len(rows), sum(r[2] for r in rows), rows[0][0]))

cum, cut, ours = 0, None, None
for i, (s, nm, n) in enumerate(rows, 1):
    cum += n
    if cut is None and cum >= 100:
        cut = (i, s, cum, nm)
    if nm == OURS:
        ours = (i, s, n, cum)

print('\n100번째 개인 컷: **%.4f**  (팀 %d위 / 누적 %d명 / %s)'
      % (cut[1], cut[0], cut[2], cut[3]))
if ours:
    print('우리 %s: %.4f | 팀 %d위 | 인원 %d | 누적 **%d번째 개인**'
          % (OURS, ours[1], ours[0], ours[2], ours[3]))
    print('격차 **%.2f**  (%d팀 / %d명 위)'
          % (cut[1] - ours[1], ours[0] - cut[0], ours[3] - cut[2]))
else:
    print('우리 팀(%s)을 못 찾았다 -- LB_TEAM 을 확인할 것' % OURS)

print('\n%-6s%12s%7s%8s' % ('팀순위', '점수', '인원', '누적'))
c = 0
for i, (s, nm, n) in enumerate(rows[:45], 1):
    c += n
    mark = ' <- 컷' if i == cut[0] else (' <- 우리' if ours and i == ours[0] else '')
    if i <= 5 or abs(i - cut[0]) <= 3 or (ours and abs(i - ours[0]) <= 2):
        print('%-6d%12.4f%7d%8d%s' % (i, s, n, c, mark))
