# 노트북에서 '아직 정의 안 된 이름' 을 쓰는 곳을 찾는다. 실행 0.
#
#   python tools/nbcheck.py experiments/v10w/aimers_v10wopt.ipynb
#
# claude.md 4-12 가 "학습 뒤쪽 셀의 사소한 NameError 로 2시간 GPU 를 날렸다" 고
# 적어두고 "눈으로 확인하는 수밖에 없다" 로 끝냈는데, 그 뒤로 같은 유형이
# **네 번** 났다 (v10wc12 / v10wmc4 / v10wph / v10wd9). 눈으로는 안 된다.
#
# 방법: 셀을 순서대로 이어붙여 한 모듈로 파싱하고, **모듈 레벨**에서 읽는 이름이
# 그보다 앞에서 묶인 적 있는지 본다. 함수 몸통 안의 참조는 호출 시점에 풀리므로
# 검사하지 않는다 (그래서 오탐이 거의 없다).
import ast
import builtins
import json
import sys

BUILTIN = set(dir(builtins)) | {"__name__", "__file__"}


def load(path):
    nb = json.load(open(path, encoding="utf-8"))
    parts, marks = [], []
    line = 1
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        parts.append(s)
        marks.append((line, i))
        line += s.count("\n") + 2
    return "\n\n".join(parts), marks


def cell_of(marks, ln):
    out = marks[0][1]
    for start, idx in marks:
        if start <= ln:
            out = idx
    return out


def bindings(tree):
    """이름 -> 처음 묶인 줄번호. 모듈 레벨과 함수 정의 모두 포함."""
    first = {}

    def put(name, ln):
        if name not in first or ln < first[name]:
            first[name] = ln

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            put(node.name, node.lineno)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            put(node.id, node.lineno)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                put((a.asname or a.name).split(".")[0], node.lineno)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            put(node.name, node.lineno)
        elif isinstance(node, (ast.arg,)):
            put(node.arg, node.lineno)
        elif isinstance(node, ast.alias):
            pass
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # STEPS_SRC / MAPPING_SRC 처럼 exec 되는 소스 문자열 안의 def 도 묶임이다
            for line in node.value.splitlines():
                t = line.strip()
                if t.startswith("def "):
                    put(t[4:].split("(")[0].strip(), node.lineno)
        elif isinstance(node, ast.comprehension):
            for t in ast.walk(node.target):
                if isinstance(t, ast.Name):
                    put(t.id, node.target.lineno)
    return first


def module_level_loads(tree):
    """함수/클래스 몸통 **바깥**에서 읽는 이름만 (줄번호와 함께)."""
    out = []
    stack = [(tree, False)]
    while stack:
        node, inside = stack.pop()
        for ch in ast.iter_child_nodes(node):
            deep = inside or isinstance(
                ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda))
            if not deep and isinstance(ch, ast.Name) and isinstance(ch.ctx, ast.Load):
                out.append((ch.id, ch.lineno))
            stack.append((ch, deep))
    return out


def check(path):
    src, marks = load(path)
    tree = ast.parse(src)
    first = bindings(tree)
    bad = []
    for name, ln in module_level_loads(tree):
        if name in BUILTIN:
            continue
        b = first.get(name)
        if b is None or b > ln:
            bad.append((ln, cell_of(marks, ln), name, b))
    seen, rows = set(), []
    for ln, ci, name, b in sorted(bad):
        if name in seen:
            continue
        seen.add(name)
        rows.append((ln, ci, name, b))
    if rows:
        print("%s -- 정의 전에 쓰는 이름 %d개" % (path, len(rows)))
        for ln, ci, name, b in rows:
            where = "정의 %s줄" % b if b else "**정의 없음**"
            print("  cell %-3d line %-5d %-28s %s" % (ci, ln, name, where))
        return 1
    print("%s -- OK (정의 전 사용 없음)" % path)
    return 0


if __name__ == "__main__":
    sys.exit(max(check(p) for p in sys.argv[1:]))
