"""Motor match-3 puro: sem captura de tela e sem mouse. Testável em qualquer SO.

Célula = (cor, tipo). cor é str (ex.: "azul" ou "0") ou None. Tipos:
  n normal | sv listrado vertical (limpa a coluna) | sh listrado horizontal (limpa a linha)
  w embrulhado (3x3) | f peixe | b bomba de cor
  x bloqueador | z inativa | u desconhecida | c vazio temporário (pós-limpeza)

Premissas NÃO verificadas no jogo real (ajuste se divergirem):
  - 4 em linha horizontal cria listrado vertical, e vice-versa (STRIPE_FROM_RUN).
  - Embrulhado limpa só 3x3 (sem a segunda explosão 5x5).
  - Peixe não tem alvo simulado: só soma bônus fixo.
  - Combos com bomba são aproximações (a escolha de orientação no jogo é aleatória).
  - Reposição aleatória não é simulada: cascatas contam só peças já visíveis (limite inferior).
  - Bloqueadores não são danificados por matches vizinhos.
"""
import random

N, SV, SH, W, F, B = "n", "sv", "sh", "w", "f", "b"
X, Z, U, C = "x", "z", "u", "c"

PIECES = {N, SV, SH, W, F, B}
COLORED = {N, SV, SH, W, F}
FIXED = {X, Z, U}

INACTIVE = (None, Z)
BLOCK = (None, X)
UNK = (None, U)
CLEAR = (None, C)

STRIPE_FROM_RUN = {"h": SV, "v": SH}

WEIGHTS = {
    "clear": 1.0,     # por célula removida
    "stripe": 3.0,    # criar listrado
    "wrapped": 4.0,   # criar embrulhado
    "bomb": 6.0,      # criar bomba de cor
    "fish": 3.0,      # peixe ativado (efeito não simulado)
    "cascade": 0.5,   # multiplicador por nível de cascata
    "low_bias": 0.02, # desempate: jogadas mais abaixo no tabuleiro
    "goal_color": None,  # cor do objetivo (ex.: "azul"); None = pontos gerais
    "goal_mult": 4.0,    # peso de cada doce da cor-alvo removido (chute, não calibrado)
}


def with_goal(color, mult=None):
    """Cópia de WEIGHTS priorizando doces de uma cor (objetivo 'coletar N doces azuis')."""
    w = dict(WEIGHTS, goal_color=color)
    if mult is not None:
        w["goal_mult"] = mult
    return w

KIND_MAP = {"normal": N, "listrado_vert": SV, "listrado_horiz": SH,
            "embrulhado": W, "peixe": F}


def parse_label(lab):
    """Rótulo do classificador (ex.: 'listrado_vert_azul') -> célula."""
    if lab == "?":
        return UNK
    if lab == "bomba":
        return (None, B)
    if lab == "bloqueador":
        return BLOCK
    if lab == "vazio":
        return INACTIVE
    t, c = lab.rsplit("_", 1)
    return (c, KIND_MAP[t]) if t in KIND_MAP else UNK


def from_color_ids(ids):
    """Matriz de índices de paleta (-1 = desconhecido) -> células normais."""
    return [[UNK if v < 0 else (str(v), N) for v in row] for row in ids]


def is_piece(cell):
    return cell[1] in PIECES


def matchable(cell):
    return cell[1] in COLORED


# ---------- detecção ----------
def find_runs(g):
    rows, cols = len(g), len(g[0])
    runs = []
    for r in range(rows):
        c = 0
        while c < cols:
            if not matchable(g[r][c]):
                c += 1
                continue
            e = c
            while e + 1 < cols and matchable(g[r][e + 1]) and g[r][e + 1][0] == g[r][c][0]:
                e += 1
            if e - c + 1 >= 3:
                runs.append(("h", [(r, j) for j in range(c, e + 1)]))
            c = e + 1
    for c in range(cols):
        r = 0
        while r < rows:
            if not matchable(g[r][c]):
                r += 1
                continue
            e = r
            while e + 1 < rows and matchable(g[e + 1][c]) and g[e + 1][c][0] == g[r][c][0]:
                e += 1
            if e - r + 1 >= 3:
                runs.append(("v", [(i, c) for i in range(r, e + 1)]))
            r = e + 1
    return runs


def merge_runs(runs):
    """Junta sequências que compartilham célula (formas L, T, +)."""
    groups = []
    for o, cells in runs:
        new = {"cells": set(cells), "runs": [(o, cells)]}
        for gp in [x for x in groups if x["cells"] & new["cells"]]:
            new["cells"] |= gp["cells"]
            new["runs"] += gp["runs"]
            groups.remove(gp)
        groups.append(new)
    return groups


def creation(group, prefer):
    """Retorna (posição, tipo) do doce especial criado pelo grupo, ou None."""
    runs = group["runs"]
    longest = max(len(c) for _, c in runs)
    if longest >= 5:
        kind = B
    elif len({o for o, _ in runs}) == 2:
        kind = W
    elif longest == 4:
        o = next(o for o, c in runs if len(c) == 4)
        kind = STRIPE_FROM_RUN[o]
    else:
        return None
    pos = next((p for p in prefer if p in group["cells"]), None) or min(group["cells"])
    return pos, kind


# ---------- efeitos ----------
def _box(g, r, c, rad):
    rows, cols = len(g), len(g[0])
    return {(i, j) for i in range(max(0, r - rad), min(rows, r + rad + 1))
            for j in range(max(0, c - rad), min(cols, c + rad + 1))}


def _row(g, r):
    return {(r, j) for j in range(len(g[0]))}


def _col(g, c):
    return {(i, c) for i in range(len(g))}


def expand(g, seeds, protected=()):
    """Remove seeds e propaga especiais. Retorna (células removidas, bônus de peixe)."""
    cleared, fish, stack = set(), 0, list(seeds)
    while stack:
        p = stack.pop()
        if p in cleared or p in protected:
            continue
        r, c = p
        kind = g[r][c][1]
        if kind in FIXED or kind == C:
            continue
        cleared.add(p)
        if kind == SV:
            stack += _col(g, c)
        elif kind == SH:
            stack += _row(g, r)
        elif kind == W:
            stack += _box(g, r, c, 1)
        elif kind == F:
            fish += 1
    return cleared, fish


def gravity(g):
    rows, cols = len(g), len(g[0])
    for c in range(cols):
        r = rows - 1
        while r >= 0:
            if g[r][c][1] in FIXED:
                r -= 1
                continue
            bottom, top = r, r
            while top - 1 >= 0 and g[top - 1][c][1] not in FIXED:
                top -= 1
            col = [g[i][c] for i in range(top, bottom + 1)]
            pieces = [x for x in col if x[1] != C]
            new = [CLEAR] * (len(col) - len(pieces)) + pieces
            for i, x in zip(range(top, bottom + 1), new):
                g[i][c] = x
            r = top - 1


def resolve(g, first_seeds=None, swapped=(), w=WEIGHTS, info=None):
    """Resolve limpezas e cascatas em g (modifica g). Retorna a pontuação.
    Se info (dict) for dado, acumula info[\"goal\"] = doces da cor-alvo removidos (só peças visíveis)."""
    score, level = 0.0, 0
    while True:
        if first_seeds is not None and level == 0:
            cleared, fish = expand(g, first_seeds)
            created = {}
        else:
            groups = merge_runs(find_runs(g))
            if not groups:
                break
            seeds, created = set(), {}
            for gp in groups:
                seeds |= gp["cells"]
                cr = creation(gp, swapped if level == 0 else ())
                if cr:
                    pos, kind = cr
                    created[pos] = (None if kind == B else g[pos[0]][pos[1]][0], kind)
            cleared, fish = expand(g, seeds - set(created), protected=set(created))
        bonus = sum({B: w["bomb"], W: w["wrapped"]}.get(k, w["stripe"]) for _, k in created.values())
        gc = w.get("goal_color")
        goal_n = sum(1 for r, c in cleared if g[r][c][0] == gc) if gc is not None else 0
        if info is not None:
            info["goal"] = info.get("goal", 0) + goal_n
        gain = (len(cleared) + goal_n * (w["goal_mult"] - 1)) * w["clear"] + bonus + fish * w["fish"]
        score += gain * (1 + w["cascade"] * level)
        for r, c in cleared:
            g[r][c] = CLEAR
        for (r, c), cell in created.items():
            g[r][c] = cell
        gravity(g)
        level += 1
        if level > 30:  # trava de segurança
            break
    return score


def _color_cells(g, color):
    return {(r, c) for r, row in enumerate(g) for c, x in enumerate(row)
            if matchable(x) and x[0] == color}


def _special_swap(g, a, b, ca, cb):
    """Seeds de combos ativados só pela troca. None se for troca comum."""
    ka, kb = ca[1], cb[1]
    if B in (ka, kb):
        if ka == kb:
            return {(r, c) for r in range(len(g)) for c in range(len(g[0])) if is_piece(g[r][c])}
        bomb, other, oc = (a, b, cb) if ka == B else (b, a, ca)
        cells = _color_cells(g, oc[0])
        seeds = {bomb, other} | cells
        if oc[1] in (SV, SH):
            for i, (r, c) in enumerate(sorted(cells)):
                seeds |= _row(g, r) if i % 2 else _col(g, c)
        elif oc[1] == W:
            for r, c in cells:
                seeds |= _box(g, r, c, 1)
        return seeds
    if ka in (SV, SH, W) and kb in (SV, SH, W):
        r, c = b
        seeds = {a, b}
        if ka == W and kb == W:
            return seeds | _box(g, r, c, 2)
        if W in (ka, kb):
            for d in (-1, 0, 1):
                if 0 <= r + d < len(g):
                    seeds |= _row(g, r + d)
                if 0 <= c + d < len(g[0]):
                    seeds |= _col(g, c + d)
            return seeds
        return seeds | _row(g, r) | _col(g, c)
    return None


def simulate(g, a, b, w=WEIGHTS, info=None):
    """Pontuação da troca a<->b, ou None se inválida."""
    ca, cb = g[a[0]][a[1]], g[b[0]][b[1]]
    if not (is_piece(ca) and is_piece(cb)) or ca == cb:
        return None
    g2 = [row[:] for row in g]
    seeds = _special_swap(g2, a, b, ca, cb)
    g2[a[0]][a[1]], g2[b[0]][b[1]] = cb, ca
    if seeds is None and not find_runs(g2):
        return None
    return resolve(g2, first_seeds=seeds, swapped=(a, b), w=w, info=info)


def rank_moves(g, w=WEIGHTS):
    """Lista [(score, a, b)] ordenada do melhor para o pior."""
    rows, cols = len(g), len(g[0])
    out = []
    for i in range(rows):
        for j in range(cols):
            for di, dj in ((0, 1), (1, 0)):
                a, b = (i, j), (i + di, j + dj)
                if b[0] >= rows or b[1] >= cols:
                    continue
                s = simulate(g, a, b, w)
                if s is not None:
                    out.append((s + w["low_bias"] * max(a[0], b[0]), a, b))
    out.sort(key=lambda t: -t[0])
    return out


def best_move(g, w=WEIGHTS):
    """Retorna ((r1,c1),(r2,c2)) ou None. Empates exatos são sorteados."""
    ranked = rank_moves(g, w)
    if not ranked:
        return None
    top = [m for m in ranked if abs(m[0] - ranked[0][0]) < 1e-9]
    _, a, b = random.choice(top)
    return a, b
