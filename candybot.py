"""Base de bot match-3 para PC (captura de tela + arrasto do mouse).

Subcomandos:
  calibrate   selecionar a área do tabuleiro e definir linhas/colunas
  learn       aprender a paleta de cores (k-means) a partir do tabuleiro atual
  show        ler o tabuleiro, imprimir a matriz e salvar debug.png
  seed        selecionar a área do jogo e rotular em lote um tabuleiro só com doces normais
  collect     (passivo) salvar recortes de células não reconhecidas enquanto você joga
  label       rotular os recortes por tecla, um a um
  verify      auditar o dataset rotulado: aponta recortes suspeitos e gera uma imagem-resumo por classe
  cluster     rotular recortes por GRUPOS de aparência parecida (bem mais rápido que label)
  train       treinar o classificador (model.npz) e mostrar acerto por classe
  kinds       ler o tabuleiro e imprimir o tipo de cada célula
  plan        ler o tabuleiro e listar as melhores jogadas com pontuação (não move o mouse)
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

import engine

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
# A busca de jogadas vive em engine.py (puro, testável sem GUI).
def read_cells(cfg, img, use_kinds=False, model=None):
    """Lê o tabuleiro como matriz de células do engine.
    Sem --kinds: só cor (paleta), tudo vira doce normal. Com --kinds: usa o classificador."""
    rows, cols = cfg["rows"], cfg["cols"]
    if use_kinds and model is not None:
        crops = [cell_img(img, rows, cols, r, c) for r in range(rows) for c in range(cols)]
        labs = predict(model, crops)
        g = [[engine.parse_label(labs[r * cols + c]) for c in range(cols)] for r in range(rows)]
        return g
    ids, _ = read_board(cfg, img)
    return engine.from_color_ids(ids)


def weights_from(args):
    gc = getattr(args, "goal_color", None)
    return engine.with_goal(gc, getattr(args, "goal_mult", None)) if gc else engine.WEIGHTS


def count_unknown(g):
    return sum(c[1] == engine.U for row in g for c in row)


def fmt_cells(g):
    return "\n".join(" ".join(f"{(c[0] or '-')[:3]:>3}:{c[1]:<2}" for c in row) for row in g)


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


def cmd_plan(args):
    """Lê o tabuleiro e mostra as melhores jogadas com pontuação. Não move o mouse."""
    cfg = load_cfg()
    model = load_model() if args.kinds else None
    if args.kinds and model is None:
        sys.exit("Sem model.npz. Rode: python candybot.py train")
    if not args.kinds and not cfg["palette"]:
        sys.exit("Paleta vazia. Rode: python candybot.py learn --k N")
    countdown(args.delay)
    g = read_cells(cfg, wait_stable(cfg["board"]), args.kinds, model)
    print(fmt_cells(g))
    print(f"{count_unknown(g)} desconhecidas de {cfg['rows'] * cfg['cols']}")
    w = weights_from(args)
    if w["goal_color"] and not any(c[0] == w["goal_color"] for row in g for c in row):
        cores = sorted({c[0] for row in g for c in row if c[0]})
        print(f"AVISO: cor-alvo '{w['goal_color']}' não existe no tabuleiro lido. Cores lidas: {cores}. "
              "Com --kinds use nomes (azul, verde...); sem --kinds, o índice da paleta.")
    ranked = engine.rank_moves(g, w)
    if not ranked:
        print("Nenhuma jogada válida encontrada.")
    for sc, a, b in ranked[:args.top]:
        extra = ""
        if w["goal_color"]:
            info = {}
            engine.simulate(g, a, b, w, info)
            extra = f"   {w['goal_color']} previstos: {info.get('goal', 0)}"
        print(f"{sc:6.2f}  {a} -> {b}{extra}")
    print("Confira a jogada 1 na tela antes de usar run.")


def cmd_run(args):
    cfg = load_cfg()
    model = load_model() if args.kinds else None
    if args.kinds and model is None:
        sys.exit("Sem model.npz. Rode: python candybot.py train")
    if not args.kinds and not cfg["palette"]:
        sys.exit("Paleta vazia. Rode: python candybot.py learn --k N")
    keyboard.add_hotkey("esc", _stop)
    print("Iniciando em 3 s. Deixe o jogo visível e em primeiro plano. ESC para parar.")
    time.sleep(3)

    w = weights_from(args)
    total = cfg["rows"] * cfg["cols"]
    moves = no_move = no_change = 0
    while not STOP and moves < args.max_moves:
        img = wait_stable(cfg["board"])
        g = read_cells(cfg, img, args.kinds, model)
        mv = engine.best_move(g, w) if count_unknown(g) <= total * args.max_unknown else None
        if mv is None:
            no_move += 1
            if no_move >= 3:
                print("Sem jogadas detectadas 3x seguidas (pop-up, fim de fase ou leitura errada). Parando.")
                break
            time.sleep(1.0)
            continue
        no_move = 0
        goal_txt = ""
        if w["goal_color"]:
            info = {}
            engine.simulate(g, *mv, w, info)
            goal_txt = f" ({w['goal_color']} previstos: {info.get('goal', 0)})"
        time.sleep(random.uniform(0.3, 0.9))
        swap(cfg, *mv)
        moves += 1
        after = read_cells(cfg, wait_stable(cfg["board"]), args.kinds, model)
        if after == g:
            no_change += 1
            print(f"Jogada {moves}: tabuleiro não mudou ({no_change}/5).")
            if no_change >= 5:
                print("Tabuleiro não reage às jogadas. Parando.")
                break
        else:
            no_change = 0
            print(f"Jogada {moves}: {mv[0]} -> {mv[1]}{goal_txt}")
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


def predict(model, crops, k=1):
    """k-NN ponderado. Retorna '?' se o vizinho mais próximo estiver longe demais.
    k=1 por padrão: classes raras (especiais com poucas amostras) perdem a votação para as normais com k>1."""
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
    t0, saved, keep, n = time.time(), 0, [], 0
    keep_sig = set()
    print(f"Coletando por {args.seconds} s. ESC ou Ctrl+C para parar; o que já foi salvo fica em {DATA / 'unlabeled'}.")
    try:
        while not STOP and time.time() - t0 < args.seconds:
            ti = time.monotonic()
            img = wait_stable(cfg["board"], min_wait=0.3, timeout=4.0)
            if time.monotonic() - ti >= 4.0:  # estourou o timeout: tela ainda animando, recorte sairia borrado
                n += 1
                print(f"captura {n}: tela instável (animação), ignorada")
                time.sleep(args.interval)
                continue
            crops = [cell_img(img, rows, cols, r, c) for r in range(rows) for c in range(cols)]
            labs = predict(model, crops) if model else ["?"] * len(crops)
            before = saved
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
            n += 1
            # time.monotonic: diagnóstico. ~5 s por captura = tela nunca estabiliza (animações do jogo).
            print(f"captura {n}: +{saved - before} novos (total {saved}), "
                  f"{labs.count('?')} '?' de {len(labs)}, {time.monotonic() - ti:.1f} s")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Interrompido.")
    print(f"{saved} recortes novos em {DATA / 'unlabeled'}. Próximo: python candybot.py label")
    if n and saved <= 2 and not args.all_cells:
        print("Quase nada salvo. Se há especiais no tabuleiro, o modelo os está lendo como doces normais "
              "(limite de desconhecido alto demais). Use: collect --seconds 10 --all-cells --dedupe 7")


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


def cluster_feats(F, dist):
    """Agrupamento guloso por líder: cada vetor entra no grupo do líder mais próximo se estiver a menos
    de dist; senão vira líder. Retorna listas de índices (líder primeiro)."""
    groups, leaders = [], []
    for i, f in enumerate(F):
        if leaders:
            d = np.linalg.norm(np.array(leaders) - f, axis=1)
            j = int(d.argmin())
            if d[j] < dist:
                groups[j].append(i)
                continue
        leaders.append(f)
        groups.append([i])
    return groups


def pick_spread(F, idxs, keep):
    """Escolhe até keep índices bem espalhados no grupo (o primeiro, depois o mais distante dos já escolhidos)."""
    chosen = [idxs[0]]
    while len(chosen) < min(keep, len(idxs)):
        rest = [i for i in idxs if i not in chosen]
        far = max(rest, key=lambda i: min(np.linalg.norm(F[i] - F[c]) for c in chosen))
        chosen.append(far)
    return chosen


def cmd_cluster(args):
    """Rotula por grupos. Cada grupo mostra até 8 exemplos; a tecla vale para o grupo todo.
    Guarda no máximo --keep amostras por grupo (as mais variadas); o resto vai para dataset/descartados."""
    files = sorted((DATA / "unlabeled").glob("*.png"))
    imgs = [(p, cv2.imread(str(p))) for p in files]
    imgs = [(p, im) for p, im in imgs if im is not None]
    if not imgs:
        sys.exit("Nada em dataset/unlabeled. Rode collect.")
    F = np.array([feat(im) for _, im in imgs], np.float32)
    groups = sorted(cluster_feats(F, args.dist), key=len, reverse=True)
    print(f"{len(imgs)} recortes em {len(groups)} grupos (dist {args.dist}). Se um grupo mostrar tipos misturados, "
          "tecle s e rode de novo com --dist menor.")
    tipos = ["tipo: " + " ".join(f"{k}={v}" for k, v in TYPES.items()),
             "sem cor: " + " ".join(f"{k}={v}" for k, v in NOCOLOR.items()) + "  s=pular  d=descartar grupo  q=sair",
             "listrado: v = listras verticais na imagem; h = horizontais"]
    (DATA / "descartados").mkdir(parents=True, exist_ok=True)

    def move(idx_list, label):
        d = DATA / label
        d.mkdir(parents=True, exist_ok=True)
        for i in idx_list:
            imgs[i][0].rename(d / imgs[i][0].name)

    for gi, idxs in enumerate(groups, 1):
        show = pick_spread(F, idxs, 8)
        head = [f"grupo {gi}/{len(groups)}: {len(idxs)} recortes. TODOS os exemplos tem o mesmo tipo e cor?"]
        k = ask(montage([imgs[i][1] for i in show]), head + tipos, set(TYPES) | set(NOCOLOR) | {"s", "q", "d"}, scale=1)
        if k == "q":
            break
        if k == "s":
            continue
        if k == "d":
            move(idxs, "descartados")
            continue
        if k in NOCOLOR:
            lab = NOCOLOR[k]
        else:
            k2 = ask(montage([imgs[i][1] for i in show]),
                     ["cor: " + " ".join(f"{a}={b}" for a, b in COLORS.items()) + "  s=pular grupo"],
                     set(COLORS) | {"s"}, scale=1)
            if k2 == "s":
                continue
            lab = f"{TYPES[k]}_{COLORS[k2]}"
        kept = pick_spread(F, idxs, args.keep)
        move(kept, lab)
        move([i for i in idxs if i not in kept], "descartados")
        print(f"grupo {gi}: {len(kept)} amostras -> {lab} ({len(idxs) - len(kept)} repetidas descartadas)")
    cv2.destroyAllWindows()


def hue_sat(crop):
    """Matiz (0-360) e saturação (0-255) da cor média do centro do recorte."""
    h, w = crop.shape[:2]
    c = crop[int(h * .25):int(h * .75), int(w * .25):int(w * .75)]
    m = c.reshape(-1, 3).mean(axis=0).astype(np.uint8).reshape(1, 1, 3)
    hsv = cv2.cvtColor(m, cv2.COLOR_BGR2HSV)[0, 0]
    return float(hsv[0]) * 2.0, float(hsv[1])


def hue_dist(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def circ_mean(hues):
    ang = np.deg2rad(hues)
    return float(np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360.0)


def audit_dataset(hue_tol=20.0):
    """Procura recortes rotulados de forma suspeita. Retorna (items, flags, class_hue).
    items = [(caminho, classe, imagem)], flags = {indice: [motivos]}.
    Heurísticas, não prova: (1) vizinho mais próximo é de outra classe; (2) longe do padrão da própria classe;
    (3) para normal_*, matiz da cor diferente do resto da classe."""
    items = []
    for d in sorted(DATA.glob("*")):
        if d.is_dir() and d.name not in ("unlabeled", "descartados"):
            for p in sorted(d.glob("*.png")):
                im = cv2.imread(str(p))
                if im is not None:
                    items.append((p, d.name, im))
    if len(items) < 2:
        return items, {i: [] for i in range(len(items))}, {}
    F = np.array([feat(im) for _, _, im in items], np.float32)
    y = np.array([lab for _, lab, _ in items])
    D = pdist(F, F)
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(axis=1)
    flags = {i: [] for i in range(len(items))}
    class_hue = {}
    for i in range(len(items)):
        if y[nn[i]] != y[i] and (y == y[i]).sum() >= 2:
            flags[i].append(f"vizinho mais próximo é {y[nn[i]]}")
    for lab in sorted(set(y)):
        idx = np.where(y == lab)[0]
        if len(idx) >= 5:
            sub = F[idx]
            Dc = pdist(sub, sub)
            med = Dc.sum(axis=1).argmin()
            d = Dc[med]
            m = float(np.median(d))
            mad = 1.4826 * float(np.median(np.abs(d - m)))
            lim = max(m + 4 * mad, 1.6 * m)
            for j, i in enumerate(idx):
                if d[j] > lim:
                    flags[i].append(f"fora do padrão da classe (dist {d[j]:.1f} > {lim:.1f})")
        if lab.startswith("normal_") and len(idx) >= 3:
            hs = [(i, *hue_sat(items[i][2])) for i in idx]
            hs = [(i, h) for i, h, sat in hs if sat >= 40]
            if len(hs) >= 3:
                ref = circ_mean(np.array([h for _, h in hs]))
                near = [h for _, h in hs if hue_dist(h, ref) < 60]
                ref = circ_mean(np.array(near)) if near else ref
                class_hue[lab] = ref
                for i, h in hs:
                    if hue_dist(h, ref) > hue_tol:
                        flags[i].append(f"matiz {h:.0f} graus, a classe está em {ref:.0f}")
    return items, flags, class_hue


def class_sheet(items, idxs, flags, cols=10, size=64):
    rows = (len(idxs) + cols - 1) // cols
    sheet = np.zeros((rows * size, cols * size, 3), np.uint8)
    for k, i in enumerate(idxs):
        r, c = divmod(k, cols)
        tile = cv2.resize(items[i][2], (size, size))
        if flags[i]:
            cv2.rectangle(tile, (0, 0), (size - 1, size - 1), (0, 0, 255), 3)
        cv2.putText(tile, str(k + 1), (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        sheet[r * size:(r + 1) * size, c * size:(c + 1) * size] = tile
    return sheet


def cmd_verify(args):
    items, flags, class_hue = audit_dataset(args.hue_tol)
    if not items:
        sys.exit("Dataset vazio. Rode seed/cluster/label antes.")
    out = Path("verificacao")
    out.mkdir(exist_ok=True)
    total = 0
    moved = []
    for lab in sorted({l for _, l, _ in items}):
        if not lab.startswith(args.only):
            continue
        idxs = [i for i, (_, l, _) in enumerate(items) if l == lab]
        bad = [i for i in idxs if flags[i]]
        total += len(bad)
        sheet = class_sheet(items, idxs, flags, args.cols)
        cv2.imwrite(str(out / f"{lab}.png"), sheet)
        aviso = "  <- poucas amostras (<5), checagem limitada" if len(idxs) < 5 else ""
        print(f"{lab:28s} n={len(idxs):3d}  suspeitas={len(bad)}{aviso}")
        for i in bad:
            print(f"    #{idxs.index(i) + 1:<3d} {items[i][0].name}: " + "; ".join(flags[i]))
        if args.show:
            cv2.imshow(lab + " (vermelho = suspeita; tecle algo)", sheet)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if args.move:
            for i in bad:
                items[i][0].rename(DATA / "unlabeled" / items[i][0].name)
                moved.append(items[i][0].name)
    if class_hue:
        print("Matiz médio por classe normal (graus; confira se os nomes fazem sentido):")
        for lab, h in sorted(class_hue.items(), key=lambda t: t[1]):
            print(f"    {lab:20s} {h:5.0f}")
        labs = sorted(class_hue, key=class_hue.get)
        for a_, b_ in zip(labs, labs[1:]):
            if hue_dist(class_hue[a_], class_hue[b_]) < args.hue_tol:
                print(f"AVISO: {a_} e {b_} têm matiz parecido; pode haver cor rotulada com o nome errado.")
    print(f"{total} suspeitas. Imagens em {out}/ (borda vermelha = suspeita; número = # impresso acima).")
    if args.move:
        print(f"{len(moved)} recortes movidos para dataset/unlabeled. Reclassifique: python candybot.py cluster --dist 3")
    elif total:
        print("Para devolver as suspeitas ao unlabeled: python candybot.py verify --move")


def cmd_train(_):
    X, y = [], []
    for d in sorted(DATA.glob("*")):
        if d.is_dir() and d.name not in ("unlabeled", "descartados"):
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
    pc.add_argument("--dedupe", type=float, default=7.0, help="distância mínima entre recortes salvos")
    pc.add_argument("--delay", type=int, default=3)
    pc.add_argument("--all-cells", action="store_true",
                    help="salva todas as células (não só '?'), com deduplicação")
    pc.set_defaults(fn=cmd_collect)
    sub.add_parser("label").set_defaults(fn=cmd_label)
    pv = sub.add_parser("verify")
    pv.add_argument("--only", default="normal_", help="prefixo das classes a reportar ('' = todas)")
    pv.add_argument("--hue-tol", type=float, default=20.0, help="tolerância de matiz em graus (classes normal_*)")
    pv.add_argument("--cols", type=int, default=10, help="colunas da imagem-resumo")
    pv.add_argument("--show", action="store_true", help="abre cada imagem-resumo numa janela")
    pv.add_argument("--move", action="store_true", help="move as suspeitas de volta para dataset/unlabeled")
    pv.set_defaults(fn=cmd_verify)
    pcl = sub.add_parser("cluster")
    pcl.add_argument("--dist", type=float, default=6.0, help="distância máxima para agrupar (menor = grupos mais puros)")
    pcl.add_argument("--keep", type=int, default=5, help="amostras guardadas por grupo")
    pcl.set_defaults(fn=cmd_cluster)
    sub.add_parser("train").set_defaults(fn=cmd_train)
    pk = sub.add_parser("kinds")
    pk.add_argument("--delay", type=int, default=3)
    pk.set_defaults(fn=cmd_kinds)
    pr = sub.add_parser("run")
    pr.add_argument("--max-moves", type=int, default=100)
    pr.add_argument("--kinds", action="store_true",
                    help="usa o classificador (model.npz) para ler doces especiais; padrão: só cor")
    pr.add_argument("--max-unknown", type=float, default=0.3,
                    help="fração máxima de células desconhecidas; acima disso não joga (pop-up/leitura ruim)")
    pr.add_argument("--goal-color", default=None,
                    help="cor do objetivo (com --kinds: azul, verde, laranja, roxo...; sem: índice da paleta)")
    pr.add_argument("--goal-mult", type=float, default=None, help="peso dos doces da cor-alvo (padrão 4)")
    pr.set_defaults(fn=cmd_run)
    pp = sub.add_parser("plan")
    pp.add_argument("--kinds", action="store_true")
    pp.add_argument("--top", type=int, default=5)
    pp.add_argument("--goal-color", default=None)
    pp.add_argument("--goal-mult", type=float, default=None)
    pp.add_argument("--delay", type=int, default=3)
    pp.set_defaults(fn=cmd_plan)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()