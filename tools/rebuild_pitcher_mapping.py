"""pitcher_id <-> pitcher_trackman_id 매핑 재구축 (로컬 실행용 CLI).

매칭 로직 자체는 여기 없다. 학습 노트북의 `MAPPING_SRC` 문자열이 단일 소스이고,
이 스크립트는 그걸 읽어 실행할 뿐이다 (STEPS_SRC <-> script.py 와 같은 패턴).
로직을 고칠 일이 있으면 노트북의 MAPPING_SRC 를 고칠 것.

왜 필요한가: 주최측이 준 pitcher_id_mapping.csv 는 구종비율 하나로만 매칭돼 약 91%가
틀렸다 (시즌 간 일관성 1.9%, 2024 커버리지 28%). 자세한 건 claude.md 2장.

출력: data/pitcher_id_mapping_v2.csv
실행: PYTHONUTF8=1 python tools/rebuild_pitcher_mapping.py
"""
import json
import os

import numpy as np
import pandas as pd

NB = 'experiments/archive/aimers_tuned_ensemble.ipynb'
OUT = 'data/pitcher_id_mapping_v2.csv'


def load_mapping_src(path=NB):
    nb = json.load(open(path, encoding='utf-8'))
    srcs = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    hits = [s for s in srcs if s.startswith('MAPPING_SRC = r"""')]
    if len(hits) != 1:
        raise RuntimeError(f'{path} 에서 MAPPING_SRC 셀을 특정하지 못했습니다 ({len(hits)}개).')
    ns = {'np': np, 'pd': pd}
    exec(hits[0].split('\nexec(MAPPING_SRC)')[0], ns)
    exec(ns['MAPPING_SRC'], ns)
    return ns['build_pitcher_map']


def main():
    build_pitcher_map = load_mapping_src()
    df_train = pd.read_csv('data/train.csv')
    df_trackman = pd.read_csv('data/trackman_history.csv')
    print(f'train {df_train.shape} | trackman {df_trackman.shape}')

    m = build_pitcher_map(df_train, df_trackman)
    os.makedirs('data', exist_ok=True)
    m.to_csv(OUT, index=False)

    cov = df_train.groupby('season').apply(
        lambda d: d.pitcher_id.isin(set(m[m.season == d.name].pitcher_id)).mean())
    print('  [커버리지] ' + '  '.join(f'{s}:{v * 100:.1f}%' for s, v in cov.items()))
    print(f'저장: {OUT} ({len(m)}행)')


if __name__ == '__main__':
    main()
