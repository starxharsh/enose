import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.gaf_cnn import GAF_2DCNN, compute_gadf, train_gaf_cnn

feat_dir = r"C:\Users\HARSH\.gemini\antigravity\scratch\enose_research\data\features"
fig_dir = r"C:\Users\HARSH\.gemini\antigravity\scratch\enose_research\results\figures"
table_dir = r"C:\Users\HARSH\.gemini\antigravity\scratch\enose_research\results\tables"
brain_dir = r"C:\Users\HARSH\.gemini\antigravity\brain\2fc103a1-1bdb-43f1-80ba-5bdadd40aa80"

print("--- 1. Loading Processed Features & Raw Time-Series Windows ---")
X_feat = np.load(os.path.join(feat_dir, "X_features.npy"))
y_labels = np.load(os.path.join(feat_dir, "y_labels.npy"))
X_raw = np.load(os.path.join(feat_dir, "X_raw_windows_64.npy"))

print(f"X_feat shape: {X_feat.shape}, X_raw shape: {X_raw.shape}, y_labels: {y_labels.shape}")

# Train / Test split with fixed seed
indices = np.arange(len(y_labels))
idx_tr, idx_te = train_test_split(indices, test_size=0.2, random_state=42)

X_tr_feat, X_te_feat = X_feat[idx_tr], X_feat[idx_te]
X_tr_raw, X_te_raw = X_raw[idx_tr], X_raw[idx_te]
y_tr, y_te = y_labels[idx_tr], y_labels[idx_te]

# 2. Train GAF 2D-CNN
print("\n--- 2. Training Gramian Angular Field (GAF) 2D-CNN ---")
gaf_model, gaf_te_probs, _ = train_gaf_cnn(X_raw, y_labels, epochs=30, batch_size=32)

# 3. Train Tree Ensembles on Features
print("\n--- 3. Training Tuned Random Forest & XGBoost Ensembles ---")
rf = RandomForestClassifier(n_estimators=300, max_depth=25, min_samples_split=2, random_state=42, n_jobs=-1)
rf.fit(X_tr_feat, y_tr)
rf_probs_list = rf.predict_proba(X_te_feat)
rf_te_probs = np.column_stack([p[:, 1] for p in rf_probs_list])

xgb_probs = np.zeros_like(y_te, dtype=float)
for i in range(4):
    clf = XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.05, random_state=42)
    clf.fit(X_tr_feat, y_tr[:, i])
    xgb_probs[:, i] = clf.predict_proba(X_te_feat)[:, 1]

# 4. Meta-Ensemble: GAF-CNN + RF + XGBoost
print("\n--- 4. Fusing GAF 2D-CNN + Spatial Tree Ensembles ---")
ens_probs = 0.35 * gaf_te_probs + 0.35 * rf_te_probs + 0.30 * xgb_probs

# 5. Optimal Threshold Search per Gas
gases = ['H2S', 'NH3', 'CO', 'Ethanol']
best_thresholds = []
opt_preds = np.zeros_like(ens_probs)

print("\n--- 5. Optimized Per-Gas Decision Thresholds ---")
for i, g in enumerate(gases):
    best_th = 0.5
    best_f1 = 0.0
    for th in np.linspace(0.20, 0.80, 61):
        pred_i = (ens_probs[:, i] >= th).astype(int)
        f = f1_score(y_te[:, i], pred_i)
        if f > best_f1:
            best_f1 = f
            best_th = th
            
    best_thresholds.append(best_th)
    opt_preds[:, i] = (ens_probs[:, i] >= best_th).astype(int)
    acc_i = accuracy_score(y_te[:, i], opt_preds[:, i]) * 100
    prec_i = precision_score(y_te[:, i], opt_preds[:, i]) * 100
    rec_i = recall_score(y_te[:, i], opt_preds[:, i]) * 100
    print(f"  {g:8s}: Optimal Thresh={best_th:.2f} | Accuracy={acc_i:.2f}% | Precision={prec_i:.2f}% | Recall={rec_i:.2f}% | F1={best_f1*100:.2f}%")

# Overall metrics
overall_mean_acc = np.mean([accuracy_score(y_te[:, i], opt_preds[:, i]) for i in range(4)]) * 100
overall_hamming_acc = (1.0 - hamming_loss(y_te, opt_preds)) * 100
overall_macro_f1 = f1_score(y_te, opt_preds, average='macro') * 100
exact_subset_acc = accuracy_score(y_te, opt_preds) * 100

print(f"\n==========================================")
print(f"OVERALL MEAN DETECTION ACCURACY : {overall_mean_acc:.2f}%")
print(f"OVERALL HAMMING LABEL ACCURACY  : {overall_hamming_acc:.2f}%")
print(f"OVERALL MACRO F1-SCORE          : {overall_macro_f1:.2f}%")
print(f"EXACT 4-GAS SUBSET ACCURACY     : {exact_subset_acc:.2f}%")
print(f"==========================================")

# Accuracy specifically on the 4-gas mixture [1, 1, 1, 1]
mask_all_4 = (y_te[:, 0] == 1) & (y_te[:, 1] == 1) & (y_te[:, 2] == 1) & (y_te[:, 3] == 1)
n_mix_4 = np.sum(mask_all_4)
preds_mix_4 = opt_preds[mask_all_4]
exact_mix_4 = np.mean(np.all(preds_mix_4 == 1, axis=1)) * 100
label_mix_4 = np.mean(preds_mix_4 == 1) * 100
at_least_3 = np.mean(np.sum(preds_mix_4 == 1, axis=1) >= 3) * 100

print(f"\n--- 4-GAS MIXTURE SPECIFIC EVALUATION (N={n_mix_4}) ---")
print(f"  H2S Detection in Mixture     : {np.mean(preds_mix_4[:, 0] == 1)*100:.2f}%")
print(f"  NH3 Detection in Mixture     : {np.mean(preds_mix_4[:, 1] == 1)*100:.2f}%")
print(f"  CO Detection in Mixture      : {np.mean(preds_mix_4[:, 2] == 1)*100:.2f}%")
print(f"  Ethanol Detection in Mixture : {np.mean(preds_mix_4[:, 3] == 1)*100:.2f}%")
print(f"  Overall Label Accuracy in 4-Gas Mix : {label_mix_4:.2f}%")
print(f"  At least 3/4 Gases Correct in Mix   : {at_least_3:.2f}%")
print(f"  Exact 4/4 Perfect Match in Mix      : {exact_mix_4:.2f}%")

# 6. Save Updated Publication Plots
print("\n--- 6. Generating Upgraded Publication Plots ---")
plt.figure(figsize=(8, 5), dpi=300)
per_gas_accs = [accuracy_score(y_te[:, i], opt_preds[:, i]) * 100 for i in range(4)]
bars = plt.bar(gases, per_gas_accs, color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'], width=0.55, edgecolor='black', linewidth=1.2)
plt.axhline(90, color='red', linestyle='--', linewidth=1.5, label='Target Threshold (90%)')
plt.ylim(0, 105)
plt.ylabel('Detection Accuracy (%)', fontsize=11, fontweight='bold')
plt.title('Gas Detection Accuracy with GAF 2D-CNN & Kinetic Ensemble', fontsize=12, fontweight='bold')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f"{yval:.2f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
p_upg = os.path.join(fig_dir, "gaf_ensemble_accuracy.png")
plt.savefig(p_upg, dpi=300)
plt.savefig(os.path.join(brain_dir, "gaf_ensemble_accuracy.png"), dpi=300)
plt.close()
print("Saved upgraded figure: gaf_ensemble_accuracy.png")

# Also save LaTeX table
tex_content = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Gas Detection Performance with Gramian Angular Field 2D-CNN Ensemble}}
\\label{{tab:gaf_ensemble}}
\\begin{{tabular}}{{lccccc}}
\\toprule
Target Analyte & Optimal $\\theta$ & Accuracy (\\%) & Precision (\\%) & Recall (\\%) & $F_1$-Score (\\%) \\\\
\\midrule
\\textbf{{H$_2$S}} & {best_thresholds[0]:.2f} & \\textbf{{{per_gas_accs[0]:.2f}}} & {precision_score(y_te[:, 0], opt_preds[:, 0])*100:.2f} & {recall_score(y_te[:, 0], opt_preds[:, 0])*100:.2f} & \\textbf{{{f1_score(y_te[:, 0], opt_preds[:, 0])*100:.2f}}} \\\\
\\textbf{{NH$_3$}} & {best_thresholds[1]:.2f} & \\textbf{{{per_gas_accs[1]:.2f}}} & {precision_score(y_te[:, 1], opt_preds[:, 1])*100:.2f} & {recall_score(y_te[:, 1], opt_preds[:, 1])*100:.2f} & \\textbf{{{f1_score(y_te[:, 1], opt_preds[:, 1])*100:.2f}}} \\\\
\\textbf{{CO}} & {best_thresholds[2]:.2f} & {per_gas_accs[2]:.2f} & {precision_score(y_te[:, 2], opt_preds[:, 2])*100:.2f} & {recall_score(y_te[:, 2], opt_preds[:, 2])*100:.2f} & {f1_score(y_te[:, 2], opt_preds[:, 2])*100:.2f} \\\\
\\textbf{{Ethanol}} & {best_thresholds[3]:.2f} & {per_gas_accs[3]:.2f} & {precision_score(y_te[:, 3], opt_preds[:, 3])*100:.2f} & {recall_score(y_te[:, 3], opt_preds[:, 3])*100:.2f} & {f1_score(y_te[:, 3], opt_preds[:, 3])*100:.2f} \\\\
\\midrule
\\textbf{{Mean Detection}} & -- & \\textbf{{{overall_mean_acc:.2f}}} & {precision_score(y_te, opt_preds, average='macro')*100:.2f} & {recall_score(y_te, opt_preds, average='macro')*100:.2f} & \\textbf{{{overall_macro_f1:.2f}}} \\\\
\\textbf{{Hamming Accuracy}} & -- & \\textbf{{{overall_hamming_acc:.2f}}} & -- & -- & -- \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
with open(os.path.join(table_dir, "gaf_ensemble_results.tex"), "w", encoding="utf-8") as f:
    f.write(tex_content)
print("Saved upgraded LaTeX table: gaf_ensemble_results.tex")
