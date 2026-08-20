# 데이콘 공식 베이스라인(RandomForest)을 우리와 '같은' 2024 홀드아웃에서 채점한다.
# 코드는 배포 노트북 그대로. 우리 v5 는 이 분할에서 784~803 이다.
#
# 왜 결정적인가: 리더보드 20~100위가 1,146~1,107 좁은 띠에 몰려 있고 제출 1~3회 팀이
# 그 안에 섞여 있다. 공식 베이스라인 조회수가 참가자 수의 2배다. 베이스라인이 1,100 대를
# 낸다면 우리(991)가 그보다 아래라는 뜻이고, 그러면 문제는 모델링이 아니라 다른 데 있다.
import os, time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

DATA_DIR = "data"
ID, TARGET = "row_id", "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]

test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                        encoding="utf-8-sig", nrows=0).columns
FEATURES = [c for c in test_cols if c != ID]
NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]
print(f"test.csv 컬럼 {len(test_cols)} -> 피처 {len(FEATURES)} "
      f"(범주 {len(CAT_COLS)} / 수치 {len(NUM_COLS)})")
print(f"season 이 피처에 포함? {'season' in FEATURES}")

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                    encoding="utf-8-sig", usecols=FEATURES + [TARGET])
print(f"train {train.shape} | 시즌 {train['season'].min()}~{train['season'].max()} "
      f"| 성공률 {train[TARGET].mean():.4f}")

pre = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
    ("num", SimpleImputer(strategy="median"), NUM_COLS),
])
model = Pipeline([("pre", pre), ("clf", RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_leaf=200, n_jobs=-1, random_state=42))])

is_val = train["season"] == 2024
Xtr, ytr = train.loc[~is_val, FEATURES], train.loc[~is_val, TARGET]
Xv, yv = train.loc[is_val, FEATURES], train.loc[is_val, TARGET]
print(f"학습 {len(Xtr):,}(~2023) -> 검증 {len(Xv):,}(2024)")

t = time.time()
model.fit(Xtr, ytr)
print(f"학습 완료 :: {time.time()-t:.1f}s")

p = model.predict_proba(Xv)[:, 1]
yv_np = yv.to_numpy()
r = yv_np.mean()
naive = r * (1 - r)
brier = ((p - yv_np) ** 2).mean()
score = max(0, 100000 * (1 - brier / naive))

print("\n" + "=" * 62)
print(f"  공식 베이스라인 2024 홀드아웃 점수 : {score:,.1f}")
print(f"  Brier {brier:.6f} | 기준선 r(1-r) {naive:.6f} | 실제 성공률 {r:.4f}")
print(f"  평균예측 {p.mean():.4f}  (실제 {r:.4f}, 편차 {p.mean()-r:+.4f})")
print(f"  예측 범위 {p.min():.4f} ~ {p.max():.4f} (std {p.std():.4f})")
print("=" * 62)
print(f"  비교: 우리 v5 는 같은 분할에서 784~803")

# 평균만 실제값에 맞춰봤을 때 (재중심화 상한) — 보정 여지가 얼마나 되는지
q = np.clip(p, 1e-6, 1 - 1e-6)
lo = np.log(q / (1 - q))
off = 0.0
for _ in range(300):
    cc = 1 / (1 + np.exp(-(lo + off)))
    e = cc.mean() - r
    if abs(e) < 1e-9:
        break
    off -= e * 4
pc = 1 / (1 + np.exp(-(lo + off)))
sc = max(0, 100000 * (1 - ((pc - yv_np) ** 2).mean() / naive))
print(f"  참고: 평균을 실제값에 맞추면 {sc:,.1f} ({sc-score:+,.1f}, 오프셋 {off:+.4f})")
