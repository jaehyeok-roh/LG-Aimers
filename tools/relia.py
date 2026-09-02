# 반쪽 분할 신뢰도 — eda33 과 **비트 단위로 같은 방식**.
#
# CAE 같은 새 지표를 만들 때 이 함수가 조용히 다르면 지표 전체가 무의미해진다.
# 그래서 구현을 한 곳에 두고 `tools/cae_selftest.py` 가 알려진 값으로 교정한다:
#
#   pitcher (커리어)   0.637      batter x season  -0.008   (claude.md eda33)
#
# eda33 과 어긋나기 쉬운 지점이 셋이고, 셋 다 여기서 고정한다.
#   ① 디트렌드가 (season, game_type) 이다. season 만 빼면 F 체제 변경이 남는다
#   ② **부모 효과를 먼저 뺀다.** batter x season 의 -0.008 은 타자 커리어 평균을
#      뺐기 때문에 나오는 값이다. 안 빼면 +0.66 이 나오고 대조군이 죽는다
#   ③ min_n 은 **반쪽마다** 40 이다. 높이면 신뢰도가 기계적으로 올라간다
import numpy as np
import pandas as pd


def add_dev(df, col='control_success', out='dev'):
    """시즌 x game_type 리그평균을 뺀 편차. 없으면 season 만으로 뺀다."""
    by = ['season', 'game_type'] if 'game_type' in df.columns else ['season']
    df[out] = df[col] - df.groupby(by)[col].transform('mean')
    return df


def reliab(df, keys, parent=None, min_n=40, val='dev', half='h'):
    """keys 그룹 편차를 무작위 반쪽 둘에서 따로 재고 상관을 본다.

    반환 (신뢰도, 셀 수, 관측std, 신호std, 커버리지).
    신호 std = 관측 std x sqrt(신뢰도)  <- 표본 잡음을 걷어낸 진짜 크기.
    """
    v = df[val].to_numpy(dtype='float64')
    if parent:
        v = v - df.groupby(parent)[val].transform('mean').to_numpy(dtype='float64')
    g = pd.DataFrame({'v': v, 'h': df[half].to_numpy()})
    for k in keys:
        g[k] = df[k].to_numpy()
    a = g.groupby(keys + ['h'])['v'].agg(['mean', 'size'])
    ok = a[a['size'] >= min_n]
    cov = float(ok['size'].sum()) / len(df)
    a = ok['mean'].unstack('h').dropna()
    if len(a) < 150:
        return None, len(a), 0.0, 0.0, cov
    r = float(np.corrcoef(a[0], a[1])[0, 1])
    obs = float(np.std(np.concatenate([a[0].to_numpy(), a[1].to_numpy()])))
    return r, len(a), obs, obs * np.sqrt(max(r, 0.0)), cov


def reliab_arr(val, grp, half, min_n=40, min_cells=30):
    """배열판. 값이 이미 편차(예: CAE 잔차)일 때 쓴다.

    `reliab` 과 같은 계산이고, 그쪽이 eda33 값을 교정으로 재현하므로
    (tools/cae_selftest.py) 이 경로도 같이 보증된다. NaN 은 버린다.
    """
    val = np.asarray(val, dtype='float64')
    ok = np.isfinite(val)
    d = pd.DataFrame({'v': val[ok], 'g': np.asarray(grp)[ok],
                      'h': np.asarray(half)[ok].astype(int)})
    a = d.groupby(['g', 'h'])['v'].agg(['mean', 'size'])
    a = a[a['size'] >= min_n]['mean'].unstack('h').dropna()
    if len(a) < min_cells or a.shape[1] < 2:
        return np.nan, np.nan, len(a)
    x, y = a.iloc[:, 0].to_numpy(), a.iloc[:, 1].to_numpy()
    r = float(np.corrcoef(x, y)[0, 1])
    obs = float(np.std(np.concatenate([x, y])))
    return r, obs * np.sqrt(max(r, 0.0)), len(a)
