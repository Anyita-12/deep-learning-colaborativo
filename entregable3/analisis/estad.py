"""Pruebas del protocolo E3: t pareada, Wilcoxon exacta, bootstrap jerárquico, Holm, efectos."""
import numpy as np
from scipy import stats

def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(1, run)
    return adj

def cliff(d):
    """delta de Cliff de las diferencias pareadas frente a 0 (proporción >0 menos <0)."""
    d = np.asarray(d); return ((d > 0).sum() - (d < 0).sum()) / len(d)

def dz(d):
    d = np.asarray(d); s = d.std(ddof=1); return d.mean() / s if s > 0 else np.inf

def t_pair(d):
    d = np.asarray(d)
    if d.std(ddof=1) == 0: return 0.0 if d.mean() != 0 else 1.0
    return stats.ttest_1samp(d, 0).pvalue

def t_nb(d, ratio=361/1800):
    """t corregida de Nadeau y Bengio."""
    d = np.asarray(d); R = len(d); v = d.var(ddof=1)
    t = d.mean() / np.sqrt((1/R + ratio) * v); return 2 * stats.t.sf(abs(t), R - 1)

def wilc(d):
    d = np.asarray(d)
    if np.all(d == 0): return 1.0
    return stats.wilcoxon(d, method="exact" if len(d) <= 25 else "auto").pvalue

def f1_w(W, y, yp, K=6, clases=None):
    """F1 macro ponderado por imagen. W: (B, n). Devuelve (B,)."""
    cl = range(K) if clases is None else clases
    f = []
    for c in cl:
        yt, pc = (y == c), (yp == c)
        tp = W @ (yt & pc); fp = W @ (~yt & pc); fn = W @ (yt & ~pc)
        den = 2*tp + fp + fn
        f.append(np.where(den > 0, 2*tp / np.where(den > 0, den, 1), 0.0))
    return np.mean(f, 0)

def _ind(y, YP, K=6):
    """Matriz (n, R*3K) con indicadores tp, fp, fn por semilla y clase."""
    cols = []
    for yp in YP:
        for c in range(K):
            yt, pc = (y == c), (yp == c)
            cols += [yt & pc, ~yt & pc, yt & ~pc]
    return np.stack(cols, 1).astype(np.float32)

def _f1_de(C, R, K=6):
    C = C.reshape(C.shape[0], R, K, 3)
    tp, fp, fn = C[..., 0], C[..., 1], C[..., 2]; den = 2*tp + fp + fn
    return np.where(den > 0, 2*tp / np.where(den > 0, den, 1), 0.0).mean(-1)   # (B, R)

def boot_jer(YA, YB, y, groups, B=10000, seed=2026, const_b=None):
    """Bootstrap pareado jerárquico: semillas y, dentro de cada una, grupos de casi duplicados.
    YA, YB: (R, n) predicciones por semilla (pareadas). const_b: F1 constante del rival.
    Devuelve (diferencias medias bootstrap (B,), p bilateral, IC95)."""
    rng = np.random.default_rng(seed)
    R, n = YA.shape
    ug, gi = np.unique(groups, return_inverse=True); G = len(ug)
    IA = _ind(y, YA); IB = None if YB is None else _ind(y, YB)
    acc = np.zeros(B); seeds = rng.integers(0, R, size=(B, R)); ar = np.arange(B)
    for j in range(R):
        cnt = rng.multinomial(G, np.full(G, 1/G), size=B).astype(np.float32)
        W = cnt[:, gi]
        fa = _f1_de(W @ IA, R)
        fb = np.full_like(fa, const_b) if IB is None else _f1_de(W @ IB, R)
        acc += fa[ar, seeds[:, j]] - fb[ar, seeds[:, j]]
    dm = acc / R
    p = max(min(1.0, 2 * min((dm <= 0).mean(), (dm >= 0).mean())), 1 / B)
    return dm, p, np.percentile(dm, [2.5, 97.5])

def boot_semillas(d, B=10000, seed=2026):
    rng = np.random.default_rng(seed); d = np.asarray(d)
    dm = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    p = max(min(1.0, 2*min((dm <= 0).mean(), (dm >= 0).mean())), 1/B)
    return dm, p, np.percentile(dm, [2.5, 97.5])

def ic_boot(f, x, B=5000, seed=7):
    rng = np.random.default_rng(seed); x = np.asarray(x)
    v = [f(x[rng.integers(0, len(x), len(x))]) for _ in range(B)]
    return np.nanpercentile(v, [2.5, 97.5])
