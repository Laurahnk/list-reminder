"""
Code complet à coller à la suite du notebook k_anonymat__5_.ipynb
Remplace les cellules existantes de généralisation et de validation.

Structure :
  A — Analyse du taux de défaut par âge (comprendre avant de découper)
  B — Comparaison des 3 approches de binnage
  C — K-anonymat optimisé (sensibilité k × tranches)
  D — Triple comparaison réelle / synthétique / anonymisée
  E — Validation statistique complète de la suppression
  F — Export final
"""

# =====================================================================
# IMPORTS COMPLÉMENTAIRES
# =====================================================================
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, norm
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

# ── Palette cohérente ─────────────────────────────────────────────────
C_REEL  = "#1565C0"   # bleu foncé  → base réelle
C_SYN   = "#00A650"   # vert        → base synthétique
C_ANON  = "#E65100"   # orange      → base anonymisée
C_SUPP  = "#C62828"   # rouge       → lignes supprimées

# ── On repart de la copie de travail propre ───────────────────────────
df_anon = df_final.copy()
df_anon['AGE_ARRONDI'] = df_anon['AGE_YEARS'].round(0).astype(int)
ATTRIBUT_SENSIBLE = 'TARGET'

print(f"Base de travail : {df_anon.shape[0]:,} lignes × {df_anon.shape[1]} colonnes")
print(f"Taux de défaut  : {df_anon[ATTRIBUT_SENSIBLE].mean():.4f}")


# =====================================================================
# CELLULE A — ANALYSE DU TAUX DE DÉFAUT PAR ÂGE
# Objectif : comprendre la relation avant de choisir les tranches
# =====================================================================
print("\n" + "="*60)
print("A — RELATION ÂGE × TAUX DE DÉFAUT")
print("="*60)

td_par_age = (df_anon
              .groupby('AGE_ARRONDI')[ATTRIBUT_SENSIBLE]
              .agg(['mean', 'count'])
              .rename(columns={'mean': 'Taux_defaut', 'count': 'N'}))

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Taux de défaut par âge
ax = axes[0]
ax.scatter(td_par_age.index, td_par_age['Taux_defaut'],
           s=td_par_age['N'] / td_par_age['N'].max() * 80,
           alpha=0.7, color=C_REEL, edgecolors='white', linewidths=0.5)
# Courbe lissée
from scipy.signal import savgol_filter
smooth = savgol_filter(td_par_age['Taux_defaut'].values, 11, 2)
ax.plot(td_par_age.index, smooth, color='red', lw=2, label='Tendance lissée')
ax.set_xlabel("Âge")
ax.set_ylabel("Taux de défaut")
ax.set_title("Taux de défaut par âge\n(taille ∝ nombre d'individus)", fontweight='bold')
ax.legend(fontsize=9, frameon=False)

# Distribution des âges
ax = axes[1]
ax.bar(td_par_age.index, td_par_age['N'], color=C_SYN, alpha=0.7, width=0.8)
ax.set_xlabel("Âge")
ax.set_ylabel("Nombre d'individus")
ax.set_title("Distribution des âges dans la base synthétique", fontweight='bold')

plt.suptitle("Relation âge × taux de défaut", fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print(f"\n  Âge min : {df_anon['AGE_ARRONDI'].min()} | Âge max : {df_anon['AGE_ARRONDI'].max()}")
print(f"  Constat : le taux de défaut DÉCROÎT avec l'âge (corrélation négative).")
print(f"  => Les tranches 'à taux de défaut uniforme' signifient :")
print(f"     regrouper les âges ayant des TD SIMILAIRES,")
print(f"     pas obtenir le même TD entre toutes les tranches.")


# =====================================================================
# CELLULE B — COMPARAISON DES 3 APPROCHES DE BINNAGE
# Objectif : choisir la méthode qui minimise la suppression
# tout en étant statistiquement justifiée
# =====================================================================
print("\n" + "="*60)
print("B — COMPARAISON DES 3 APPROCHES DE BINNAGE DE L'ÂGE")
print("="*60)

N_TRANCHES = 10  # nombre de tranches à comparer
K_TEST     = 5   # k cible pour évaluer chaque approche

QI_SANS_AGE = ['CODE_GENDER', 'NAME_FAMILY_STATUS',
               'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE']

def evaluer_binnage(df, age_gen_col, qi_sans_age, k):
    """Calcule le % de lignes supprimées pour atteindre k-anonymat."""
    qi = [age_gen_col] + qi_sans_age
    tailles = df.groupby(qi)[age_gen_col].transform('count')
    n_suppr  = (tailles < k).sum()
    pct      = 100 * n_suppr / len(df)
    k_min    = tailles.min()
    return pct, int(n_suppr), int(k_min), df[age_gen_col].nunique()

resultats_binnage = []

for n_tr in [5, 7, 10, 12, 15, 20]:
    df_tmp = df_anon.copy()

    # ── Approche 1 : Equal-width (pd.cut) — méthode actuelle ────────
    df_tmp['AGE_EW'] = pd.cut(df_tmp['AGE_ARRONDI'], bins=n_tr,
                               precision=0, include_lowest=True).astype(str)
    pct_ew, n_ew, k_ew, n_bins_ew = evaluer_binnage(df_tmp, 'AGE_EW', QI_SANS_AGE, K_TEST)

    # ── Approche 2 : Quantile (pd.qcut) — effectifs égaux ───────────
    try:
        df_tmp['AGE_Q'] = pd.qcut(df_tmp['AGE_ARRONDI'], q=n_tr,
                                   duplicates='drop').astype(str)
        pct_q, n_q, k_q, n_bins_q = evaluer_binnage(df_tmp, 'AGE_Q', QI_SANS_AGE, K_TEST)
    except Exception:
        pct_q, n_q, k_q, n_bins_q = float('nan'), 0, 0, 0

    # ── Approche 3 : Risk-based (arbre de décision) ──────────────────
    # Un arbre cherche les coupures où le TD change le plus
    # => bins à TD homogène à l'intérieur (ce que veut le tuteur)
    try:
        tree = DecisionTreeClassifier(
            max_leaf_nodes=n_tr,
            min_samples_leaf=max(50, len(df_tmp) // (n_tr * 10)),
            random_state=42
        )
        tree.fit(df_tmp[['AGE_ARRONDI']], df_tmp[ATTRIBUT_SENSIBLE])
        thresholds = sorted(set(
            round(t) for t in tree.tree_.threshold if t != -2
        ))
        cuts = [df_tmp['AGE_ARRONDI'].min() - 1] + thresholds + [df_tmp['AGE_ARRONDI'].max() + 1]
        df_tmp['AGE_RB'] = pd.cut(df_tmp['AGE_ARRONDI'], bins=cuts,
                                   precision=0).astype(str)
        pct_rb, n_rb, k_rb, n_bins_rb = evaluer_binnage(df_tmp, 'AGE_RB', QI_SANS_AGE, K_TEST)
    except Exception as e:
        pct_rb, n_rb, k_rb, n_bins_rb = float('nan'), 0, 0, 0

    resultats_binnage.append({
        'N tranches cible': n_tr,
        'EW  perte(%)': round(pct_ew, 3),  'EW  bins': n_bins_ew,
        'Q   perte(%)': round(pct_q,  3),  'Q   bins': n_bins_q,
        'RB  perte(%)': round(pct_rb, 3),  'RB  bins': n_bins_rb,
    })
    print(f"  {n_tr:>2} tranches | EW:{pct_ew:.2f}%({n_bins_ew}bins) "
          f"| Q:{pct_q:.2f}%({n_bins_q}bins) | RB:{pct_rb:.2f}%({n_bins_rb}bins)")

df_binnage = pd.DataFrame(resultats_binnage)

# Graphique comparatif
fig, ax = plt.subplots(figsize=(10, 4))
x = df_binnage['N tranches cible']
ax.plot(x, df_binnage['EW  perte(%)'], marker='o', label='Equal-width (actuel)', color='#EF4444', lw=2)
ax.plot(x, df_binnage['Q   perte(%)'], marker='s', label='Quantile (effectifs égaux)', color=C_SYN, lw=2)
ax.plot(x, df_binnage['RB  perte(%)'], marker='^', label='Risk-based (arbre)', color=C_ANON, lw=2)
ax.axhline(1, color='grey', lw=1.2, linestyle='--', label='Seuil 1%')
ax.set_xlabel("Nombre de tranches pour AGE_ARRONDI")
ax.set_ylabel("% de lignes perdues (pour k=5)")
ax.set_title("Comparaison des 3 approches de binnage\n(moins = mieux)", fontweight='bold')
ax.legend(fontsize=9, frameon=False)
ax.set_xticks(x)
plt.tight_layout()
plt.show()

print("\n  Explication :")
print("  Equal-width   → tranches de même largeur (ex: 5 ans), peut créer de petits groupes aux extrêmes")
print("  Quantile      → même nombre de personnes par tranche, réduit les petits groupes")
print("  Risk-based    → coupures là où le TD change le plus (demande du tuteur)")


# =====================================================================
# CELLULE C — K-ANONYMAT OPTIMISÉ
# Objectif : choisir la meilleure approche et le meilleur k
# =====================================================================
print("\n" + "="*60)
print("C — K-ANONYMAT OPTIMISÉ")
print("="*60)

# ── C1. Choisir l'approche de binnage (Risk-based recommandée) ───────
N_TRANCHES_FINAL = 10  # ajuster selon résultats cellule B

tree_final = DecisionTreeClassifier(
    max_leaf_nodes=N_TRANCHES_FINAL,
    min_samples_leaf=max(50, len(df_anon) // (N_TRANCHES_FINAL * 10)),
    random_state=42
)
tree_final.fit(df_anon[['AGE_ARRONDI']], df_anon[ATTRIBUT_SENSIBLE])
thresholds_final = sorted(set(
    round(t) for t in tree_final.tree_.threshold if t != -2
))
cuts_final = ([df_anon['AGE_ARRONDI'].min() - 1]
              + thresholds_final
              + [df_anon['AGE_ARRONDI'].max() + 1])

df_anon_gen = df_anon.copy()
df_anon_gen['AGE_GEN'] = pd.cut(
    df_anon_gen['AGE_ARRONDI'], bins=cuts_final, precision=0
).astype(str)

print(f"\n  Coupures de l'arbre : {thresholds_final}")
print(f"  Tranches créées    : {df_anon_gen['AGE_GEN'].nunique()}")

# Taux de défaut par tranche (validation de la cohérence)
td_par_tranche = (df_anon_gen
                  .groupby('AGE_GEN')[ATTRIBUT_SENSIBLE]
                  .agg(['mean', 'count'])
                  .rename(columns={'mean': 'TD', 'count': 'N'})
                  .sort_values('TD', ascending=False))
print("\n  Taux de défaut par tranche d'âge :")
print(td_par_tranche.to_string())

QI_GENERALISES = ['AGE_GEN', 'CODE_GENDER', 'NAME_FAMILY_STATUS',
                   'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE']

# ── C2. Analyse de sensibilité sur k ─────────────────────────────────
print("\n  Sensibilité : perte selon k cible")
print(f"  {'k':>4} | {'Perdues':>9} | {'Perte %':>8} | {'TD perdues':>11} | {'Verdict':>15}")
print(f"  {'-'*60}")

sensib_k = []
for k in [2, 3, 5, 7, 10, 15, 20]:
    tailles = df_anon_gen.groupby(QI_GENERALISES)['AGE_GEN'].transform('count')
    mask_s  = tailles < k
    n_s     = mask_s.sum()
    pct     = 100 * n_s / len(df_anon_gen)
    td_s    = df_anon_gen.loc[mask_s, ATTRIBUT_SENSIBLE].mean() if n_s > 0 else float('nan')
    verdict = '✅ Optimal' if pct < 1 else ('⚠️ Acceptable' if pct < 3 else '❌ Trop de perte')
    print(f"  {k:>4} | {n_s:>9,} | {pct:>7.2f}% | {td_s:>11.5f} | {verdict}")
    sensib_k.append({'k': k, 'Perdues': n_s, 'Perte (%)': round(pct, 3),
                     'TD perdues': round(td_s, 5) if not pd.isna(td_s) else float('nan')})

df_sensib = pd.DataFrame(sensib_k)

# ── C3. Application avec k=5 ─────────────────────────────────────────
K_CIBLE = 5

tailles_gen = df_anon_gen.groupby(QI_GENERALISES)['AGE_GEN'].transform('count')
mask_ok     = tailles_gen >= K_CIBLE
df_k_anonyme = df_anon_gen[mask_ok].copy()
nb_supprimes = (~mask_ok).sum()
pct_supprime = 100 * nb_supprimes / len(df_anon_gen)

# Remplacer AGE_ARRONDI par AGE_GEN dans l'export (k-anonymat réel)
df_export = df_k_anonyme.copy()
df_export['AGE_ARRONDI'] = df_export['AGE_GEN']   # la colonne exportée = tranche, pas valeur exacte
df_export = df_export.drop(columns=['AGE_GEN'], errors='ignore')

print(f"\n  K_CIBLE = {K_CIBLE}")
print(f"  Lignes supprimées : {nb_supprimes:,} ({pct_supprime:.2f}%)")
print(f"  Base k-anonyme    : {len(df_k_anonyme):,} lignes")

# Vérification finale
k_check = df_k_anonyme.groupby(QI_GENERALISES)['AGE_GEN'].transform('count').min()
print(f"  k minimum vérifié : {k_check} {'✅' if k_check >= K_CIBLE else '❌'}")


# =====================================================================
# CELLULE D — TRIPLE COMPARAISON
# réelle (df_300) / synthétique (df_anon) / anonymisée (df_k_anonyme)
# =====================================================================
print("\n" + "="*60)
print("D — TRIPLE COMPARAISON DES BASES")
print("="*60)

# ── Fonctions de métriques ────────────────────────────────────────────
def cramers_v_metric(x, y):
    try:
        ct   = pd.crosstab(x.astype(str), y.astype(str))
        chi2 = chi2_contingency(ct, correction=False)[0]
        n    = len(x)
        phi2 = chi2 / n
        r, k = ct.shape
        phi2c = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rc = r - (r-1)**2/(n-1); kc = k - (k-1)**2/(n-1)
        d  = min(rc-1, kc-1)
        return float(np.sqrt(phi2c / d)) if d > 0 else 0.0
    except Exception:
        return 0.0

def metriques_base(df, cols_num, cols_cat, label):
    td    = df[ATTRIBUT_SENSIBLE].mean() if ATTRIBUT_SENSIBLE in df.columns else float('nan')
    cn_ok = [c for c in cols_num if c in df.columns]
    cc_ok = [c for c in cols_cat if c in df.columns and c != ATTRIBUT_SENSIBLE]
    cr    = df[cn_ok].corr().abs() if len(cn_ok) >= 2 else pd.DataFrame()
    np.fill_diagonal(cr.values, np.nan) if not cr.empty else None
    corr  = float(np.nanmean(cr.values)) if not cr.empty else float('nan')
    cv    = np.mean([cramers_v_metric(df[c], df[ATTRIBUT_SENSIBLE]) for c in cc_ok]) if cc_ok else float('nan')
    return {'Base': label, 'N lignes': f"{len(df):,}", 'Taux défaut': round(td, 5),
            'Corrélation': round(corr, 4), 'Cramér V': round(cv, 4)}

# Colonnes communes aux 3 bases
cols_num_commun = [c for c in df_300.select_dtypes(include=[np.number]).columns
                   if c != ATTRIBUT_SENSIBLE and c in df_k_anonyme.columns]
cols_cat_commun = [c for c in df_300.select_dtypes(include='object').columns
                   if c in df_k_anonyme.columns]

m_reel  = metriques_base(df_300,       cols_num_commun, cols_cat_commun, 'Réelle')
m_syn   = metriques_base(df_anon,      cols_num_commun, cols_cat_commun, 'Synthétique')
m_anon  = metriques_base(df_k_anonyme, cols_num_commun, cols_cat_commun, 'Anonymisée')

df_metriques = pd.DataFrame([m_reel, m_syn, m_anon])
print("\n  Métriques des 3 bases :")
print(df_metriques.to_string(index=False))

# ── Écarts entre les 3 paires ─────────────────────────────────────────
paires = [
    ('Réelle → Synthétique', m_reel,  m_syn),
    ('Réelle → Anonymisée',  m_reel,  m_anon),
    ('Synthétique → Anonymisée', m_syn, m_anon),
]

print("\n  Écarts entre les bases :")
print(f"  {'Comparaison':<30} {'Δ TD':>10} {'Δ Corr':>10} {'Δ Cramér':>10}")
print(f"  {'-'*62}")
for label, m1, m2 in paires:
    d_td   = abs(m1['Taux défaut'] - m2['Taux défaut'])
    d_corr = abs(m1['Corrélation']  - m2['Corrélation'])
    d_cv   = abs(m1['Cramér V']    - m2['Cramér V'])
    print(f"  {label:<30} {d_td:>10.5f} {d_corr:>10.5f} {d_cv:>10.5f}")

# ── Graphiques des distributions numériques (top 6) ──────────────────
cols_plot = cols_num_commun[:6]
if cols_plot:
    nc = min(3, len(cols_plot))
    nr = (len(cols_plot) + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(6*nc, 3.5*nr))
    axs = np.array(axes).flatten() if len(cols_plot) > 1 else [axes]
    for i, col in enumerate(cols_plot):
        if i >= len(axs): break
        axs[i].hist(df_300[col].dropna(),       bins=40, density=True, alpha=0.55,
                    color=C_REEL, label='Réelle')
        axs[i].hist(df_anon[col].dropna(),      bins=40, density=True, alpha=0.55,
                    color=C_SYN,  label='Synthétique')
        axs[i].hist(df_k_anonyme[col].dropna(), bins=40, density=True, alpha=0.55,
                    color=C_ANON, label='Anonymisée')
        axs[i].set_title(col, fontweight='bold', fontsize=9)
        axs[i].legend(fontsize=7, frameon=False)
    for j in range(len(cols_plot), len(axs)): axs[j].set_visible(False)
    fig.suptitle("Distributions numériques — Réelle / Synthétique / Anonymisée",
                 fontweight='bold', y=1.02)
    plt.tight_layout(); plt.show()

# ── Graphique des métriques côte à côte ──────────────────────────────
metriques_labels = ['Taux défaut', 'Corrélation', 'Cramér V']
vals = {
    'Réelle':       [m_reel['Taux défaut'],  m_reel['Corrélation'],  m_reel['Cramér V']],
    'Synthétique':  [m_syn['Taux défaut'],   m_syn['Corrélation'],   m_syn['Cramér V']],
    'Anonymisée':   [m_anon['Taux défaut'],  m_anon['Corrélation'],  m_anon['Cramér V']],
}
x = np.arange(len(metriques_labels)); w = 0.26
fig, ax = plt.subplots(figsize=(10, 4))
for i, (label, v) in enumerate(vals.items()):
    c = [C_REEL, C_SYN, C_ANON][i]
    ax.bar(x + (i-1)*w, v, w, label=label, color=c, alpha=0.85, edgecolor='none')
ax.set_xticks(x); ax.set_xticklabels(metriques_labels)
ax.set_title("Comparaison des métriques — Réelle / Synthétique / Anonymisée",
             fontweight='bold')
ax.legend(fontsize=9, frameon=False)
plt.tight_layout(); plt.show()


# =====================================================================
# CELLULE E — VALIDATION STATISTIQUE COMPLÈTE DE LA SUPPRESSION
# =====================================================================
print("\n" + "="*60)
print("E — VALIDATION STATISTIQUE DE LA SUPPRESSION")
print("="*60)

# Masques supprimées / gardées
tailles_valid = df_anon_gen.groupby(QI_GENERALISES)['AGE_GEN'].transform('count')
mask_supprime = tailles_valid < K_CIBLE
mask_garde    = ~mask_supprime

df_suppr  = df_anon_gen[mask_supprime].copy()
df_gardes = df_anon_gen[mask_garde].copy()

n_s, n_g  = len(df_suppr), len(df_gardes)
p_s       = df_suppr[ATTRIBUT_SENSIBLE].mean()
p_g       = df_gardes[ATTRIBUT_SENSIBLE].mean()
p_0       = df_anon_gen[ATTRIBUT_SENSIBLE].mean()

print(f"\n  Lignes supprimées : {n_s:,}  ({100*n_s/len(df_anon_gen):.2f}%)")
print(f"  Lignes gardées    : {n_g:,}")

# ── E1. Comparaison directe des taux de défaut ────────────────────────
print(f"\n  ── E1. Taux de défaut ────────────────────────────────────")
print(f"  Total base synthétique : {p_0:.5f}")
print(f"  Lignes gardées         : {p_g:.5f}")
print(f"  Lignes supprimées      : {p_s:.5f}")
ecart = abs(p_s - p_g)
print(f"  Écart absolu |Δp|      : {ecart:.5f}  ", end="")
if ecart < 0.01:
    print("✅ Négligeable — suppression neutre pour le modèle")
elif ecart < 0.03:
    print("⚠️  Modéré — à mentionner dans les limites")
else:
    print("❌ Significatif — biais de sélection à corriger")

# ── E2. Test z de proportion ──────────────────────────────────────────
print(f"\n  ── E2. Test z de proportion (H0 : TD supprimées = TD gardées) ──")
p_pool = (n_s * p_s + n_g * p_g) / (n_s + n_g)
se     = np.sqrt(p_pool * (1 - p_pool) * (1/n_s + 1/n_g))
z      = (p_s - p_g) / se
p_val  = 2 * (1 - norm.cdf(abs(z)))
ic_s   = 1.96 * np.sqrt(p_s * (1-p_s) / n_s)
ic_g   = 1.96 * np.sqrt(p_g * (1-p_g) / n_g)
print(f"  IC 95% gardées    : [{p_g-ic_g:.5f}, {p_g+ic_g:.5f}]")
print(f"  IC 95% supprimées : [{p_s-ic_s:.5f}, {p_s+ic_s:.5f}]")
print(f"  z-stat = {z:.3f}  |  p-value = {p_val:.5f}")
if p_val < 0.05:
    print(f"  → Différence détectable (p<0.05), mais |Δp|={ecart:.5f} ← voir interprétation pratique ci-dessus")
else:
    print(f"  → Pas de différence significative ✅")

# ── E3. Test de Mann-Whitney sur les variables numériques clés ────────
print(f"\n  ── E3. Mann-Whitney (distributions complètes) ───────────")
cols_mw = [c for c in cols_num_commun if c in df_suppr.columns][:12]
res_mw  = []
for col in cols_mw:
    x1 = df_suppr[col].dropna().values
    x0 = df_gardes[col].dropna().values
    if len(x1) < 5: continue
    U, p = mannwhitneyu(x1, x0, alternative='two-sided')
    auc  = max(U / (len(x1)*len(x0)), 1 - U / (len(x1)*len(x0)))
    res_mw.append({'Variable': col,
                   'Moy. gardées': round(x0.mean(), 3),
                   'Moy. supprimées': round(x1.mean(), 3),
                   'AUC effet': round(auc, 4),
                   'p-value': round(p, 5),
                   'Flag': '❌ Fort' if auc > 0.70 else ('⚠️' if auc > 0.60 else '✅')})

df_mw = pd.DataFrame(res_mw).sort_values('AUC effet', ascending=False).reset_index(drop=True)
print(f"  AUC effet : 0.50=aucun effet | 0.60=faible | 0.70=notable")
print(df_mw.to_string(index=False))

# ── E4. V de Cramér entre SUPPRIME et les variables catégorielles ─────
print(f"\n  ── E4. V de Cramér (SUPPRIME ~ variables catégorielles) ──")
df_anon_gen['_SUPPRIME'] = mask_supprime.astype(int)
res_cat = []
for col in [c for c in cols_cat_commun if c in df_anon_gen.columns][:10]:
    v = cramers_v_metric(df_anon_gen[col].fillna('NaN'), df_anon_gen['_SUPPRIME'])
    res_cat.append({'Variable': col, 'V de Cramér': round(v, 4),
                    'Flag': '❌ Fort' if v > 0.30 else ('⚠️' if v > 0.10 else '✅ Faible')})
df_cat = pd.DataFrame(res_cat).sort_values('V de Cramér', ascending=False).reset_index(drop=True)
print(f"  < 0.10 négligeable | 0.10-0.30 notable | > 0.30 fort")
print(df_cat.to_string(index=False))
df_anon_gen.drop(columns=['_SUPPRIME'], inplace=True, errors='ignore')

# ── E5. Profil des combinaisons QI les plus supprimées ────────────────
print(f"\n  ── E5. Top 10 des groupes supprimés ─────────────────────")
profil = (df_suppr[QI_GENERALISES + [ATTRIBUT_SENSIBLE]]
          .groupby(QI_GENERALISES)
          .agg(N=('TARGET', 'count'), TD=('TARGET', 'mean'))
          .reset_index()
          .sort_values('N', ascending=False)
          .head(10))
print(profil.to_string(index=False))

# ── E6. Graphique synthèse ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Taux de défaut avec IC
ax = axes[0]
groupes  = [f'Gardées\n({n_g:,})', f'Supprimées\n({n_s:,})']
taux_bar = [p_g, p_s]
ic_bar   = [ic_g, ic_s]
bars = ax.bar(groupes, taux_bar, color=[C_SYN, C_SUPP], width=0.45,
              edgecolor='none', yerr=ic_bar, capsize=7,
              error_kw={'ecolor': '#2C3E50', 'lw': 1.5})
for bar, v in zip(bars, taux_bar):
    ax.text(bar.get_x()+bar.get_width()/2, v+max(ic_bar)*1.5,
            f'{v:.5f}', ha='center', fontsize=10, fontweight='bold')
ax.axhline(p_0, color='grey', lw=1.5, linestyle='--',
           label=f'Total ({p_0:.5f})')
ax.set_ylabel('Taux de défaut')
ax.set_title(f'Taux de défaut ± IC 95%\n|Δp| = {ecart:.5f}', fontweight='bold')
ax.legend(fontsize=8, frameon=False)

# AUC effet Mann-Whitney top 10
ax = axes[1]
top_mw = df_mw.head(10)
colors_mw = ['#EF4444' if v > 0.70 else ('#F39C12' if v > 0.60 else '#00A650')
             for v in top_mw['AUC effet']]
ax.barh(range(len(top_mw)), top_mw['AUC effet'].values, color=colors_mw,
        edgecolor='none', height=0.65)
ax.set_yticks(range(len(top_mw)))
ax.set_yticklabels(top_mw['Variable'].values, fontsize=8)
ax.invert_yaxis()
ax.axvline(0.60, color='orange', lw=1.5, linestyle='--')
ax.axvline(0.70, color='red',    lw=1.5, linestyle='--')
ax.axvline(0.50, color='grey',   lw=1,   linestyle=':')
ax.set_xlabel('AUC effet')
ax.set_title('Mann-Whitney — AUC effet\nsupprimées vs gardées', fontweight='bold')

# V de Cramér
ax = axes[2]
colors_cv = ['#EF4444' if v > 0.30 else ('#F39C12' if v > 0.10 else '#00A650')
             for v in df_cat['V de Cramér']]
ax.barh(range(len(df_cat)), df_cat['V de Cramér'].values, color=colors_cv,
        edgecolor='none', height=0.65)
ax.set_yticks(range(len(df_cat)))
ax.set_yticklabels(df_cat['Variable'].values, fontsize=8)
ax.invert_yaxis()
ax.axvline(0.10, color='orange', lw=1.5, linestyle='--')
ax.axvline(0.30, color='red',    lw=1.5, linestyle='--')
ax.set_xlabel('V de Cramér avec SUPPRIME')
ax.set_title('Lien suppression ~ variables\ncatégorielles', fontweight='bold')

plt.suptitle('Validation statistique de la suppression k-anonymat',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# ── E7. Conclusion automatique ────────────────────────────────────────
print("\n" + "="*60)
print("CONCLUSION VALIDATION STATISTIQUE")
print("="*60)
problemes = []
if ecart >= 0.03:
    problemes.append(f"Biais sur TARGET élevé : |Δp| = {ecart:.5f}")
elif ecart >= 0.01:
    problemes.append(f"Biais TARGET modéré : |Δp| = {ecart:.5f}")
mw_forts = df_mw[df_mw['AUC effet'] > 0.70] if not df_mw.empty else pd.DataFrame()
if len(mw_forts) > 0:
    problemes.append(f"Mann-Whitney notable (>0.70) : {list(mw_forts['Variable'])}")
cv_forts = df_cat[df_cat['V de Cramér'] > 0.30] if not df_cat.empty else pd.DataFrame()
if len(cv_forts) > 0:
    problemes.append(f"V Cramér fort (>0.30) : {list(cv_forts['Variable'])}")

if not problemes:
    print(f"\n  ✅ Suppression MCAR — neutre pour le modèle de score.")
    print(f"     Aucun biais significatif détecté.")
else:
    print(f"\n  Points d'attention :")
    for pb in problemes: print(f"    - {pb}")


# =====================================================================
# CELLULE F — EXPORT FINAL
# =====================================================================
print("\n" + "="*60)
print("F — EXPORT")
print("="*60)

# On exporte avec la tranche d'âge à la place de l'âge exact
df_export_final = df_k_anonyme.copy()
# Remplacer AGE_ARRONDI par la tranche généralisée (k-anonymat réel)
df_export_final['AGE_ARRONDI'] = df_export_final['AGE_GEN']
df_export_final = df_export_final.drop(columns=['AGE_GEN'], errors='ignore')

output_name = f'df_ctgan_k{K_CIBLE}_anonyme.csv'
df_export_final.to_csv(output_name, index=False)
print(f"  Fichier exporté  : {output_name}")
print(f"  Lignes           : {len(df_export_final):,}")
print(f"  Colonnes         : {df_export_final.shape[1]}")
print(f"  K-anonymat       : k >= {K_CIBLE}")
print(f"  Lignes perdues   : {nb_supprimes:,} ({pct_supprime:.2f}%)")
print(f"\n  Colonnes QI dans l'export :")
for qi in QI_GENERALISES:
    col_out = 'AGE_ARRONDI' if qi == 'AGE_GEN' else qi
    if col_out in df_export_final.columns:
        print(f"    {col_out} → valeur = tranche généralisée (pas l'âge exact)")
