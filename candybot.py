"""Base de bot match-3 para PC (captura de tela + arrasto do mouse).

Subcomandos:
  calibrate   selecionar a área do tabuleiro e definir linhas/colunas
  learn       aprender a paleta de cores (k-means) a partir do tabuleiro atual
  show        ler o tabuleiro, imprimir a matriz e salvar debug.png
  seed        selecionar a área do jogo e rotular em lote um tabuleiro só com doces normais
  collect     (passivo) salvar recortes de células não reconhecidas enquanto você joga
  label       rotular os recortes por tecla
  train       treinar o classificador (model.npz) e mostrar acerto por classe
  kinds       ler o tabuleiro e imprimir o tipo de cada célula
  run         executar o bot (ESC para parar; mouse no canto superior esquerdo aborta)
"""
import argparse
import ctypes
import json
import random
import sys
import time
from pathlib import Path

import cv2
import keyboard
import mss
import numpy as np
import pyautogui

try:
    ctypes.windll.user32.SetProcessDPIAware()
except AttributeError:
    pass  # fora do Windows

CONFIG = Path("config.json")
MAX_COLOR_DIST = 60.0   # acima disso a célula vira "desconhecida" (-1)
STABLE_DIFF = 1.5       # diferença média de pixel para considerar a tela parada
STOP = False

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.02
SCT = mss.MSS() if hasattr(mss, "MSS") else mss.mss()


# ---------- config ----------
def load_cfg():
    if not CONFIG.exists():
        sys.exit("config.json não existe. Rode: python candybot.py calibrate")
    return json.loads(CONFIG.read_text())


def save_cfg(cfg):
    CONFIG.write_text(json.dumps(cfg, indent=2))


# ---------- captura ----------
def grab(board):
    return np.ascontiguousarray(np.array(SCT.grab(board))[:, :, :3])  # BGR


def wait_stable(board, min_wait=0.4, timeout=6.0):
    """Espera animações/cascatas terminarem (2 frames seguidos quase iguais)."""
    time.sleep(min_wait)
    prev, ok, t0 = grab(board), 0, time.time()
    while time.time() - t0 < timeout:
        time.sleep(0.2)
        cur = grab(board)
        diff = float(cv2.absdiff(cur, prev).mean())
        ok = ok + 1 if diff < STABLE_DIFF else 0
        prev = cur
        if ok >= 2:
            return cur
    return prev


# ---------- leitura do tabuleiro ----------
def cell_colors(img, rows, cols):
    h, w = img.shape[:2]
    ch, cw = h / rows, w / cols
    out = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = int((r + .3) * ch), int((r + .7) * ch)
            x0, x1 = int((c + .3) * cw), int((c + .7) * cw)
            out.append(img[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0))
    return np.array(out, dtype=np.float32)


def classify(colors, palette, rows, cols, radius=None):
    pal = np.array(palette, dtype=np.float32)
    d = np.linalg.norm(colors[:, None, :] - pal[None, :, :], axis=2)
    lab = d.argmin(axis=1)
    dmin = d.min(axis=1)
    lim = np.array(radius, dtype=np.float32)[lab] if radius else MAX_COLOR_DIST
    lab[dmin > lim] = -1
    return lab.reshape(rows, cols).tolist()


def read_board(cfg, img=None):
    img = grab(cfg["board"]) if img is None else img
    return classify(cell_colors(img, cfg["rows"], cfg["cols"]),
                    cfg["palette"], cfg["rows"], cfg["cols"], cfg.get("radius")), img


# ---------- lógica ----------
def find_matches(g):
    n, m = len(g), len(g[0])
    hit = set()
    for i in range(n):
        for j in range(m):
            c = g[i][j]
            if c < 0:
                continue
            if j + 2 < m and g[i][j + 1] == c and g[i][j + 2] == c:
                hit |= {(i, j), (i, j + 1), (i, j + 2)}
            if i + 2 < n and g[i + 1][j] == c and g[i + 2][j] == c:
                hit |= {(i, j), (i + 1, j), (i + 2, j)}
    return hit


def best_move(g):
    n, m = len(g), len(g[0])
    best, best_score = [], 0
    for i in range(n):
        for j in range(m):
            for di, dj in ((0, 1), (1, 0)):
                a, b = i + di, j + dj
                if a >= n or b >= m or g[i][j] < 0 or g[a][b] < 0:
                    continue
                g[i][j], g[a][b] = g[a][b], g[i][j]
                s = len(find_matches(g))
                g[i][j], g[a][b] = g[a][b], g[i][j]
                if s > best_score:
                    best, best_score = [((i, j), (a, b))], s
                elif s == best_score and s > 0:
                    best.append(((i, j), (a, b)))
    return random.choice(best) if best else None


# ---------- ação ----------
def cell_center(cfg, r, c):
    b = cfg["board"]
    cw, ch = b["width"] / cfg["cols"], b["height"] / cfg["rows"]
    return b["left"] + (c + .5) * cw, b["top"] + (r + .5) * ch


def swap(cfg, a, b):
    x1, y1 = cell_center(cfg, *a)
    x2, y2 = cell_center(cfg, *b)
    pyautogui.moveTo(x1, y1, duration=random.uniform(0.05, 0.15))
    pyautogui.dragTo(x2, y2, duration=random.uniform(0.12, 0.25), button="left")


# ---------- comandos ----------
def select_board(monitor=1, delay=0):
    """Congela a tela e deixa você desenhar o retângulo do tabuleiro (funciona com tela dividida).
    Retorna (board, recorte BGR do tabuleiro)."""
    countdown(delay)
    mon = SCT.monitors[monitor]
    img = np.ascontiguousarray(np.array(SCT.grab(mon))[:, :, :3])
    scale = min(1.0, 1400 / img.shape[1], 800 / img.shape[0])
    small = cv2.resize(img, None, fx=scale, fy=scale)
    x, y, w, h = cv2.selectROI("Arraste sobre as celulas do tabuleiro e tecle ENTER", small, showCrosshair=False)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    if w == 0 or h == 0:
        sys.exit("Seleção cancelada.")
    x0, y0, x1, y1 = (int(v / scale) for v in (x, y, x + w, y + h))
    board = {"left": mon["left"] + x0, "top": mon["top"] + y0, "width": x1 - x0, "height": y1 - y0}
    return board, img[y0:y1, x0:x1].copy()


def cmd_calibrate(_):
    board, _img = select_board()
    rows = int(input("Linhas do tabuleiro: "))
    cols = int(input("Colunas do tabuleiro: "))
    save_cfg({"board": board, "rows": rows, "cols": cols, "palette": []})
    print("Salvo em config.json. Próximo passo: python candybot.py learn --k 6")


def countdown(sec):
    if sec <= 0:
        return
    print(f"Capturando em {sec} s. Deixe o tabuleiro visível, sem o terminal sobre ele.")
    time.sleep(sec)


def cmd_learn(args):
    cfg = load_cfg()
    countdown(args.delay)
    img = grab(cfg["board"])
    cv2.imwrite("learn.png", img)  # o que o bot realmente enxergou
    colors = cell_colors(img, cfg["rows"], cfg["cols"])
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, labels, centers = cv2.kmeans(colors, args.k, None, crit, 10, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    radius = []
    for i in range(args.k):
        m = colors[labels == i]
        r = float(np.linalg.norm(m - centers[i], axis=1).max()) if len(m) else 0.0
        radius.append(max(25.0, 2.0 * r))
        print(f"cor {i}: {len(m)} células, dispersão máx {r:.1f}")
    cfg["palette"] = centers.tolist()
    cfg["radius"] = radius
    save_cfg(cfg)
    print("Paleta salva. Abra learn.png para conferir a captura e rode: python candybot.py show")


def cmd_show(args):
    cfg = load_cfg()
    if not cfg["palette"]:
        sys.exit("Paleta vazia. Rode: python candybot.py learn --k N")
    countdown(args.delay)
    g, img = read_board(cfg)
    for row in g:
        print(" ".join(f"{v:2d}" if v >= 0 else " ?" for v in row))
    unk = sum(v < 0 for row in g for v in row)
    if unk > cfg["rows"] * cfg["cols"] // 2:
        print(f"{unk} células desconhecidas. Provável: a paleta foi aprendida com outra imagem "
              "na tela. Rode learn de novo com o tabuleiro visível e compare learn.png com debug.png.")
    h, w = img.shape[:2]
    ch, cw = h / cfg["rows"], w / cfg["cols"]
    for r in range(cfg["rows"]):
        for c in range(cfg["cols"]):
            cv2.rectangle(img, (int(c * cw), int(r * ch)),
                          (int((c + 1) * cw), int((r + 1) * ch)), (255, 255, 255), 1)
            cv2.putText(img, str(g[r][c]), (int(c * cw + cw * .35), int(r * ch + ch * .6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.imwrite("debug.png", img)
    print("debug.png salvo. Se a matriz não bate com a tela, ajuste calibrate/learn.")


def _stop():
    global STOP
    STOP = True


def cmd_run(args):
    cfg = load_cfg()
    if not cfg["palette"]:
        sys.exit("Paleta vazia. Rode: python candybot.py learn --k N")
    keyboard.add_hotkey("esc", _stop)
    print("Iniciando em 3 s. Deixe o jogo visível e em primeiro plano. ESC para parar.")
    time.sleep(3)

    moves = no_move = no_change = 0
    while not STOP and moves < args.max_moves:
        img = wait_stable(cfg["board"])
        g, _ = read_board(cfg, img)
        mv = best_move(g)
        if mv is None:
            no_move += 1
            if no_move >= 3:
                print("Sem jogadas detectadas 3x seguidas (pop-up, fim de fase ou leitura errada). Parando.")
                break
            time.sleep(1.0)
            continue
        no_move = 0
        time.sleep(random.uniform(0.3, 0.9))
        swap(cfg, *mv)
        moves += 1
        after, _ = read_board(cfg, wait_stable(cfg["board"]))
        if after == g:
            no_change += 1
            print(f"Jogada {moves}: tabuleiro não mudou ({no_change}/5).")
            if no_change >= 5:
                print("Tabuleiro não reage às jogadas. Parando.")
                break
        else:
            no_change = 0
            print(f"Jogada {moves}: {mv[0]} -> {mv[1]}")
    print(f"Fim. {moves} jogadas.")


# ---------- aprendizado: dataset e classificador ----------
DATA = Path("dataset")
MODEL = Path("model.npz")
FEAT = 24  # recorte reduzido para FEAT x FEAT
COLORS = {"1": "vermelho", "2": "laranja", "3": "amarelo", "4": "verde", "5": "azul", "6": "roxo"}
TYPES = {"n": "normal", "v": "listrado_vert", "h": "listrado_horiz", "e": "embrulhado", "p": "peixe"}
NOCOLOR = {"b": "bomba", "x": "bloqueador", "z": "vazio"}
ABBR = {"normal": "n", "listrado_vert": "lv", "listrado_horiz": "lh", "embrulhado": "e", "peixe": "p"}


def cell_img(img, rows, cols, r, c, m=0.04):
    h, w = img.shape[:2]
    ch, cw = h / rows, w / cols
    return img[int((r + m) * ch):int((r + 1 - m) * ch), int((c + m) * cw):int((c + 1 - m) * cw)]


def feat(crop):
    """Pixels (BGR) + magnitude de borda, em FEAT x FEAT. Borda ajuda a ver listras/embalagem."""
    s = cv2.resize(crop, (FEAT, FEAT), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ed = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)) / 1000.0
    return np.concatenate([s.astype(np.float32).ravel() / 255.0, 2.0 * ed.ravel()]).astype(np.float32)


def pdist(A, B):
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.sqrt(np.maximum(d2, 0))


def load_model():
    if not MODEL.exists():
        return None
    z = np.load(MODEL, allow_pickle=False)
    return {"X": z["X"], "y": z["y"], "thr": float(z["thr"])}


def predict(model, crops, k=3):
    """k-NN ponderado. Retorna '?' se o vizinho mais próximo estiver longe demais."""
    D = pdist(np.array([feat(c) for c in crops]), model["X"])
    out = []
    for d in D:
        idx = np.argsort(d)[:k]
        votes = {}
        for i in idx:
            votes[str(model["y"][i])] = votes.get(str(model["y"][i]), 0) + 1 / (d[i] + 1e-6)
        out.append(max(votes, key=votes.get) if d[idx[0]] <= model["thr"] else "?")
    return out


def short(lab):
    if lab == "?":
        return "?"
    if lab in NOCOLOR.values():
        return {"bomba": "BB", "bloqueador": "XX", "vazio": ".."}[lab]
    t, c = lab.rsplit("_", 1)
    return ABBR[t] + c[:2]


def ask(crop, lines, valid, scale=4):
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    h, w = big.shape[:2]
    canvas = np.zeros((h + 22 * len(lines) + 8, max(w, 760), 3), np.uint8)
    canvas[:h, :w] = big
    for i, t in enumerate(lines):  # ASCII apenas: o OpenCV não desenha acentos
        cv2.putText(canvas, t, (5, h + 22 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    while True:
        cv2.imshow("rotular", canvas)
        k = chr(cv2.waitKey(0) & 0xFF)
        if k in valid:
            return k


def _save(crop, label, tag):
    d = DATA / label
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / f"{tag}_{time.time_ns()}.png"), crop)


def fit_palette(colors, k):
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, labels, centers = cv2.kmeans(colors, k, None, crit, 10, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    radius = []
    for i in range(k):
        m = colors[labels == i]
        r = float(np.linalg.norm(m - centers[i], axis=1).max()) if len(m) else 0.0
        radius.append(max(25.0, 2.0 * r))
    return centers.tolist(), radius, labels


def montage(crops, n=8, size=64):
    return np.hstack([cv2.resize(c, (size, size)) for c in crops[:n]])


def cmd_seed(args):
    """Você seleciona a área do jogo; o script agrupa as células por cor e você nomeia cada grupo.
    Só use com um tabuleiro sem especiais nem bloqueadores."""
    cfg = load_cfg()
    board, img = select_board(args.monitor, args.delay)
    cfg["board"] = board
    cfg["rows"] = args.rows or cfg["rows"]
    cfg["cols"] = args.cols or cfg["cols"]
    rows, cols = cfg["rows"], cfg["cols"]
    k = args.k or len(cfg.get("palette") or []) or 5
    cfg["palette"], cfg["radius"], labels = fit_palette(cell_colors(img, rows, cols), k)
    save_cfg(cfg)
    cv2.imwrite("learn.png", img)
    print(f"Área salva em config.json: {board}. Grade {rows}x{cols}. Paleta reaprendida com k={k}.")
    help_ = ["grupo: confira se TODOS os doces mostrados tem a mesma cor",
             "cor: " + " ".join(f"{a}={b}" for a, b in COLORS.items()),
             "s=pular (grupo misturado: rode de novo com --k maior)"]
    for i in range(k):
        cells = [int(j) for j in np.where(labels == i)[0]]
        if not cells:
            continue
        crops = [cell_img(img, rows, cols, j // cols, j % cols) for j in cells]
        key = ask(montage(crops), help_ + [f"grupo {i + 1}/{k}: {len(cells)} celulas"], set(COLORS) | {"s"}, scale=1)
        if key == "s":
            print(f"grupo {i}: pulado")
            continue
        for c in crops:
            _save(c, f"normal_{COLORS[key]}", "seed")
        print(f"grupo {i}: {len(cells)} recortes -> normal_{COLORS[key]}")
    cv2.destroyAllWindows()


def cmd_collect(args):
    """Passivo: você joga à mão e salva recortes novos em unlabeled. Não move o mouse."""
    cfg = load_cfg()
    rows, cols = cfg["rows"], cfg["cols"]
    model = load_model()
    (DATA / "unlabeled").mkdir(parents=True, exist_ok=True)
    keyboard.add_hotkey("esc", _stop)
    countdown(args.delay)
    t0, saved, keep = time.time(), 0, []
    keep_sig = set()
    while not STOP and time.time() - t0 < args.seconds:
        img = wait_stable(cfg["board"], min_wait=0.3, timeout=4.0)
        crops = [cell_img(img, rows, cols, r, c) for r in range(rows) for c in range(cols)]
        labs = predict(model, crops) if model else ["?"] * len(crops)
        for crop, lab in zip(crops, labs):
            if not args.all_cells and lab != "?":
                continue
            # Assinatura visual curta para descartar duplicatas quase idênticas rapidamente.
            sig = cv2.resize(crop, (8, 8), interpolation=cv2.INTER_AREA)
            sig = (sig.astype(np.uint8) // 16).tobytes()
            if sig in keep_sig:
                continue
            f = feat(crop)
            if any(np.linalg.norm(f - o) < args.dedupe for o in keep):
                continue
            keep.append(f)
            keep_sig.add(sig)
            _save(crop, "unlabeled", "c")
            saved += 1
        time.sleep(args.interval)
    print(f"{saved} recortes novos em {DATA / 'unlabeled'}. Próximo: python candybot.py label")


def cmd_label(_):
    files = sorted((DATA / "unlabeled").glob("*.png"))
    if not files:
        sys.exit("Nada em dataset/unlabeled. Rode collect.")
    tipos = ["tipo: " + " ".join(f"{k}={v}" for k, v in TYPES.items()),
             "sem cor: " + " ".join(f"{k}={v}" for k, v in NOCOLOR.items()) + "  s=pular q=sair",
             "listrado: vert = listras verticais na imagem; horiz = listras horizontais"]
    for n, p in enumerate(files, 1):
        im = cv2.imread(str(p))
        k = ask(im, [f"{n}/{len(files)}"] + tipos, set(TYPES) | set(NOCOLOR) | {"s", "q"})
        if k == "q":
            break
        if k == "s":
            continue
        if k in NOCOLOR:
            lab = NOCOLOR[k]
        else:
            k2 = ask(im, ["cor: " + " ".join(f"{a}={b}" for a, b in COLORS.items()) + "  s=pular"], set(COLORS) | {"s"})
            if k2 == "s":
                continue
            lab = f"{TYPES[k]}_{COLORS[k2]}"
        (DATA / lab).mkdir(parents=True, exist_ok=True)
        p.rename(DATA / lab / p.name)
    cv2.destroyAllWindows()


def cmd_train(_):
    X, y = [], []
    for d in sorted(DATA.glob("*")):
        if d.is_dir() and d.name != "unlabeled":
            for p in d.glob("*.png"):
                im = cv2.imread(str(p))
                if im is not None:
                    X.append(feat(im))
                    y.append(d.name)
    if len(X) < 2:
        sys.exit("Dataset vazio. Rode seed/collect/label antes.")
    X, y = np.array(X, np.float32), np.array(y)
    D = pdist(X, X)
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(axis=1)
    ok = y[nn] == y
    print(f"{len(X)} amostras, {len(set(y))} classes. Acerto leave-one-out (1-NN): {ok.mean():.1%}")
    for c in sorted(set(y)):
        m = y == c
        aviso = "  <- poucas amostras (<5)" if m.sum() < 5 else ""
        print(f"  {c:28s} n={m.sum():3d}  acerto={ok[m].mean():.0%}{aviso}")
    for i in np.where(~ok)[0][:15]:
        print(f"  erro: {y[i]} lido como {y[nn[i]]}")
    nd = D.min(axis=1)
    thr = 1.5 * float(np.percentile(nd[ok], 95)) if ok.any() else float(nd[np.isfinite(nd)].max())
    np.savez(MODEL, X=X, y=y, thr=thr)
    print(f"model.npz salvo (limite de desconhecido = {thr:.2f}).")


def cmd_kinds(args):
    cfg = load_cfg()
    model = load_model()
    if model is None:
        sys.exit("Sem model.npz. Rode: python candybot.py train")
    countdown(args.delay)
    img = grab(cfg["board"])
    rows, cols = cfg["rows"], cfg["cols"]
    labs = predict(model, [cell_img(img, rows, cols, r, c) for r in range(rows) for c in range(cols)])
    for r in range(rows):
        print(" ".join(f"{short(l):>4}" for l in labs[r * cols:(r + 1) * cols]))
    print(f"{labs.count('?')} desconhecidas de {len(labs)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("calibrate").set_defaults(fn=cmd_calibrate)
    pl = sub.add_parser("learn")
    pl.add_argument("--k", type=int, default=6, help="número de cores de doce")
    pl.add_argument("--delay", type=int, default=3, help="segundos antes de capturar")
    pl.set_defaults(fn=cmd_learn)
    ps = sub.add_parser("show")
    ps.add_argument("--delay", type=int, default=3, help="segundos antes de capturar")
    ps.set_defaults(fn=cmd_show)
    pse = sub.add_parser("seed")
    pse.add_argument("--delay", type=int, default=3, help="segundos antes de congelar a tela")
    pse.add_argument("--monitor", type=int, default=1, help="índice do monitor do mss (1 = principal)")
    pse.add_argument("--k", type=int, default=None, help="número de cores (padrão: tamanho da paleta atual)")
    pse.add_argument("--rows", type=int, default=None)
    pse.add_argument("--cols", type=int, default=None)
    pse.set_defaults(fn=cmd_seed)
    pc = sub.add_parser("collect")
    pc.add_argument("--seconds", type=int, default=120, help="duração da coleta")
    pc.add_argument("--interval", type=float, default=1.0, help="pausa entre capturas")
    pc.add_argument("--dedupe", type=float, default=2.0, help="distância mínima entre recortes salvos")
    pc.add_argument("--delay", type=int, default=3)
    pc.add_argument("--all-cells", action="store_true",
                    help="salva todas as células (não só '?'), com deduplicação")
    pc.set_defaults(fn=cmd_collect)
    sub.add_parser("label").set_defaults(fn=cmd_label)
    sub.add_parser("train").set_defaults(fn=cmd_train)
    pk = sub.add_parser("kinds")
    pk.add_argument("--delay", type=int, default=3)
    pk.set_defaults(fn=cmd_kinds)
    pr = sub.add_parser("run")
    pr.add_argument("--max-moves", type=int, default=100)
    pr.set_defaults(fn=cmd_run)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()