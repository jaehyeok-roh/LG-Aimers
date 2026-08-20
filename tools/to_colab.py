# 캐글용 노트북 -> 코랩용으로 변환.
#
# 코랩 런타임 종류는 UI 메뉴가 아니라 **노트북 메타데이터**에 저장된다.
#   metadata.accelerator = "GPU",  metadata.colab.gpuType = "T4"
# 이걸 파일에 박아두면 코랩이 열 때 GPU 런타임으로 잡으므로 메뉴를 누를 필요가 없다.
#
# 추가로 맨 앞에 셀 두 개를 넣는다:
#   1) Drive 마운트 + catboost 설치 (코랩엔 기본 설치가 아니다)
#   2) GPU 가드 — 없으면 즉시 중단. 조용히 CPU 로 돌다가 몇 시간 태우는 사고를 막는다.
#
# 사용: python to_colab.py <in.ipynb> [<out.ipynb>]
import json, os, sys

SETUP = '''# --- 코랩 환경 준비 (캐글에서는 자동으로 건너뜀) ---
import os, subprocess, sys
IS_COLAB = os.path.isdir("/content") and not os.path.isdir("/kaggle")
if IS_COLAB:
    if not os.path.isdir("/content/drive/MyDrive"):
        from google.colab import drive
        drive.mount("/content/drive")
    try:
        import catboost  # noqa: F401
    except ImportError:
        print("catboost 설치 중...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "catboost"], check=True)
    print("코랩 준비 완료", flush=True)
'''

GUARD = '''# --- GPU 가드: 없으면 여기서 멈춘다 ---
# CatBoost 는 task_type='GPU' 인데 GPU 가 없으면 예외를 던지지 않고 느리게 돌거나
# 뒤늦게 죽는다. 몇 시간 태우기 전에 여기서 끊는다.
import subprocess
_r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                     "--format=csv,noheader"], capture_output=True, text=True)
if _r.returncode != 0:
    raise RuntimeError(
        "GPU 가 없다. 코랩이면 런타임 > 런타임 유형 변경 > T4 GPU 로 바꿀 것.\\n"
        + (_r.stderr or "")[:300])
print("GPU:", _r.stdout.strip(), flush=True)
'''


def cell(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


def main():
    if len(sys.argv) < 2:
        sys.exit("사용: python to_colab.py <in.ipynb> [<out.ipynb>]")
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src.replace(".ipynb", "_colab.ipynb")
    nb = json.load(open(src, encoding="utf-8"))

    md = nb.setdefault("metadata", {})
    md["accelerator"] = "GPU"
    md["colab"] = {"provenance": [], "gpuType": "T4", "toc_visible": True}
    md.setdefault("kernelspec", {"name": "python3", "display_name": "Python 3"})

    if not any("IS_COLAB" in "".join(c["source"]) for c in nb["cells"]):
        nb["cells"] = [cell(SETUP), cell(GUARD)] + nb["cells"]

    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None

    import ast
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    json.dump(nb, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{dst} 생성 — accelerator=GPU(T4), 셀 {len(nb['cells'])}개, 문법 OK")


if __name__ == "__main__":
    main()
