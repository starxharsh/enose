"""
Evaluation metrics and IEEE-quality result generation for e-nose research.
Produces confusion matrices, ROC curves, comparison tables (LaTeX), and publication-ready plots.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score,
    hamming_loss, cohen_kappa_score, matthews_corrcoef,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, multilabel_confusion_matrix
)
import os
import warnings
warnings.filterwarnings('ignore')

# IEEE-style plot settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

GAS_NAMES = ['H₂S', 'NH₃', 'CO', 'C₂H₅OH']
GAS_CODES = ['h', 'n', 'c', 'e']
COLORS = sns.color_palette('tab10', 13)


def compute_classification_metrics(y_true, y_pred, y_prob=None):
    """Compute comprehensive classification metrics for multi-label predictions."""
    results = {}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Overall metrics
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['hamming_loss'] = hamming_loss(y_true, y_pred)
    results['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    results['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    results['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    results['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)

    # Per-gas metrics
    for i, gas in enumerate(GAS_NAMES):
        results[f'{gas}_precision'] = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        results[f'{gas}_recall'] = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        results[f'{gas}_f1'] = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        if y_prob is not None:
            try:
                results[f'{gas}_auc'] = roc_auc_score(y_true[:, i], y_prob[:, i])
            except ValueError:
                results[f'{gas}_auc'] = float('nan')

    # MCC per gas (averaged)
    mccs = []
    for i in range(y_true.shape[1]):
        try:
            mccs.append(matthews_corrcoef(y_true[:, i], y_pred[:, i]))
        except Exception:
            mccs.append(0.0)
    results['mcc_mean'] = np.mean(mccs)

    return results


def compute_regression_metrics(y_true, y_pred, gas_names=None):
    """Compute comprehensive regression metrics."""
    results = {}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    if gas_names is None:
        gas_names = [f'Gas_{i}' for i in range(y_true.shape[1])]

    for i, gas in enumerate(gas_names):
        yt, yp = y_true[:, i], y_pred[:, i]
        results[f'{gas}_r2'] = r2_score(yt, yp)
        results[f'{gas}_rmse'] = np.sqrt(mean_squared_error(yt, yp))
        results[f'{gas}_mae'] = mean_absolute_error(yt, yp)
        # MAPE (avoid division by zero)
        mask = yt != 0
        if mask.sum() > 0:
            results[f'{gas}_mape'] = np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100
        else:
            results[f'{gas}_mape'] = float('nan')

    # Overall averages
    results['r2_mean'] = np.mean([results[f'{g}_r2'] for g in gas_names])
    results['rmse_mean'] = np.mean([results[f'{g}_rmse'] for g in gas_names])
    results['mae_mean'] = np.mean([results[f'{g}_mae'] for g in gas_names])

    return results


def plot_confusion_matrices(y_true, y_pred, model_name, save_dir):
    """Plot per-gas confusion matrices in a 1x4 grid."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    for i, (ax, gas) in enumerate(zip(axes, GAS_NAMES)):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Absent', 'Present'],
                    yticklabels=['Absent', 'Present'])
        ax.set_title(gas)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    fig.suptitle(f'Confusion Matrices — {model_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'cm_{model_name.lower().replace(" ", "_")}.png'))
    plt.savefig(os.path.join(save_dir, f'cm_{model_name.lower().replace(" ", "_")}.pdf'))
    plt.close()


def plot_roc_curves(y_true, y_probs_dict, save_dir):
    """Plot ROC curves for all models, per gas."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    y_true = np.array(y_true)

    for i, (ax, gas) in enumerate(zip(axes, GAS_NAMES)):
        for model_name, y_prob in y_probs_dict.items():
            y_prob = np.array(y_prob)
            if y_prob.ndim < 2 or y_prob.shape[1] <= i:
                continue
            try:
                fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
                auc_val = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'{model_name} ({auc_val:.3f})')
            except Exception:
                continue
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_title(gas)
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend(fontsize=6, loc='lower right')

    fig.suptitle('ROC Curves per Gas', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curves_all.png'))
    plt.savefig(os.path.join(save_dir, 'roc_curves_all.pdf'))
    plt.close()


def plot_model_comparison_bar(results_dict, metric_keys, save_dir, filename='model_comparison'):
    """Grouped bar chart comparing models across metrics."""
    models = list(results_dict.keys())
    n_metrics = len(metric_keys)
    x = np.arange(len(models))
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(7.16, 4))
    for j, metric in enumerate(metric_keys):
        values = [results_dict[m].get(metric, 0) for m in models]
        bars = ax.bar(x + j * width - 0.4 + width/2, values, width, label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{filename}.png'))
    plt.savefig(os.path.join(save_dir, f'{filename}.pdf'))
    plt.close()


def plot_predicted_vs_actual(y_true, y_pred, gas_names, model_name, save_dir):
    """Scatter plots of predicted vs actual concentration."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if y_true.ndim == 1:
        y_true, y_pred = y_true.reshape(-1, 1), y_pred.reshape(-1, 1)

    n_gases = y_true.shape[1]
    fig, axes = plt.subplots(1, n_gases, figsize=(3.5 * n_gases, 3.5))
    if n_gases == 1:
        axes = [axes]

    for i, (ax, gas) in enumerate(zip(axes, gas_names)):
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=15, c=COLORS[i])
        mn, mx = y_true[:, i].min(), y_true[:, i].max()
        ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.5, label='Ideal')
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        ax.set_title(f'{gas} (R²={r2:.4f})')
        ax.set_xlabel('Actual (ppm)')
        ax.set_ylabel('Predicted (ppm)')
        ax.legend(fontsize=8)

    fig.suptitle(f'Predicted vs Actual — {model_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'pred_vs_actual_{model_name.lower().replace(" ", "_")}.png'))
    plt.savefig(os.path.join(save_dir, f'pred_vs_actual_{model_name.lower().replace(" ", "_")}.pdf'))
    plt.close()


def plot_radar_chart(results_dict, metric_keys, save_dir):
    """Radar/spider chart for multi-metric comparison."""
    models = list(results_dict.keys())
    N = len(metric_keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for idx, model in enumerate(models):
        values = [results_dict[model].get(m, 0) for m in metric_keys]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=1.5, label=model, color=COLORS[idx % len(COLORS)])
        ax.fill(angles, values, alpha=0.08, color=COLORS[idx % len(COLORS)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_keys, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=7)
    ax.set_title('Multi-Metric Model Comparison', fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'radar_comparison.png'))
    plt.savefig(os.path.join(save_dir, 'radar_comparison.pdf'))
    plt.close()


def generate_latex_table(results_dict, metric_keys, caption, label, save_path):
    """Generate a LaTeX booktabs-style table."""
    models = list(results_dict.keys())
    header = ' & '.join(['Model'] + [m.replace('_', ' ').title() for m in metric_keys])

    lines = [
        '\\begin{table}[htbp]',
        '\\centering',
        f'\\caption{{{caption}}}',
        f'\\label{{{label}}}',
        '\\begin{tabular}{l' + 'c' * len(metric_keys) + '}',
        '\\toprule',
        header + ' \\\\',
        '\\midrule',
    ]

    # Find best values per column for bolding
    best_vals = {}
    for m in metric_keys:
        vals = [results_dict[model].get(m, 0) for model in models]
        if 'loss' in m.lower() or 'rmse' in m.lower() or 'mae' in m.lower() or 'mape' in m.lower() or 'hamming' in m.lower():
            best_vals[m] = min(vals) if vals else 0
        else:
            best_vals[m] = max(vals) if vals else 0

    for model in models:
        row_vals = []
        for m in metric_keys:
            val = results_dict[model].get(m, 0)
            if isinstance(val, (int, float)) and not np.isnan(val):
                formatted = f'{val:.4f}'
                if m in best_vals and abs(val - best_vals[m]) < 1e-6:
                    formatted = f'\\textbf{{{formatted}}}'
            else:
                formatted = 'N/A'
            row_vals.append(formatted)
        lines.append(f'{model} & ' + ' & '.join(row_vals) + ' \\\\')

    lines += [
        '\\bottomrule',
        '\\end{tabular}',
        '\\end{table}',
    ]

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  LaTeX table saved: {save_path}")


def generate_all_results(clf_results, reg_results, y_true_clf, y_preds_clf,
                         y_probs_clf, y_true_reg, y_preds_reg, save_dir):
    """Generate all IEEE-quality figures and tables."""
    fig_dir = os.path.join(save_dir, 'figures')
    tab_dir = os.path.join(save_dir, 'tables')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    print("\n=== Generating IEEE Paper Results ===\n")

    # 1. Confusion matrices for each model
    print("[1/7] Confusion matrices...")
    for model_name, y_pred in y_preds_clf.items():
        plot_confusion_matrices(y_true_clf, y_pred, model_name, fig_dir)

    # 2. ROC curves
    print("[2/7] ROC curves...")
    if y_probs_clf:
        plot_roc_curves(y_true_clf, y_probs_clf, fig_dir)

    # 3. Classification comparison bar chart
    print("[3/7] Classification comparison...")
    clf_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    plot_model_comparison_bar(clf_results, clf_metrics, fig_dir, 'clf_comparison')

    # 4. Radar chart
    print("[4/7] Radar chart...")
    radar_metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'mcc_mean']
    plot_radar_chart(clf_results, radar_metrics, fig_dir)

    # 5. Predicted vs actual for regression
    print("[5/7] Regression scatter plots...")
    for model_name, y_pred in y_preds_reg.items():
        gas_names_reg = list(reg_results.get(model_name, {}).keys())
        gas_names_reg = [g.replace('_r2', '') for g in gas_names_reg if g.endswith('_r2')]
        if not gas_names_reg:
            gas_names_reg = ['CO']
        plot_predicted_vs_actual(y_true_reg, y_pred, gas_names_reg, model_name, fig_dir)

    # 6. LaTeX tables
    print("[6/7] LaTeX tables...")
    generate_latex_table(
        clf_results,
        ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'hamming_loss', 'mcc_mean'],
        'Classification Performance Comparison',
        'tab:classification',
        os.path.join(tab_dir, 'classification_comparison.tex')
    )

    if reg_results:
        reg_metrics = ['r2_mean', 'rmse_mean', 'mae_mean']
        generate_latex_table(
            reg_results, reg_metrics,
            'Regression Performance Comparison',
            'tab:regression',
            os.path.join(tab_dir, 'regression_comparison.tex')
        )

    # 7. Per-gas classification table
    print("[7/7] Per-gas tables...")
    for gas in GAS_NAMES:
        gas_metrics = [f'{gas}_precision', f'{gas}_recall', f'{gas}_f1']
        available = {m: r for m, r in clf_results.items()
                     if all(k in r for k in gas_metrics)}
        if available:
            generate_latex_table(
                available, gas_metrics,
                f'Classification Performance for {gas}',
                f'tab:{gas.lower()}_clf',
                os.path.join(tab_dir, f'{gas.lower().replace("₂", "2").replace("₅", "5")}_classification.tex')
            )

    print("\n=== All results generated! ===")
    print(f"  Figures: {fig_dir}")
    print(f"  Tables:  {tab_dir}")
