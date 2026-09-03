"""
E-Nose Research Pipeline — Main Orchestration Script
=====================================================
Runs the complete pipeline: Load → Preprocess → Features → Explore → Train → Evaluate → Results

Usage:
    python pipeline.py                    # Run full pipeline
    python pipeline.py --phase preprocess # Run specific phase
    python pipeline.py --skip-uci        # Skip UCI datasets (faster)
"""

import sys
import os
import time
import json
import pickle
import warnings
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# Project root
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data'
RESULTS_DIR = PROJECT_DIR / 'results'
CUSTOM_DATA_DIR = Path(r'C:\Users\HARSH\Downloads\Telegram Desktop')

sys.path.insert(0, str(PROJECT_DIR / 'src'))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def phase_preprocess():
    """Phase 1 & 2: Load data, preprocess, extract features."""
    print("\n" + "=" * 70)
    print("  PHASE 1 & 2: DATA LOADING, PREPROCESSING & FEATURE EXTRACTION")
    print("=" * 70)

    from preprocessing.load_custom_data import load_custom_dataset, SENSOR_COLS
    from preprocessing.feature_extraction import extract_all_features

    # --- Load custom dataset ---
    print("\n[1/4] Loading custom dataset...")
    t0 = time.time()
    dataset = load_custom_dataset(str(CUSTOM_DATA_DIR))
    print(f"  Loaded in {time.time()-t0:.1f}s")

    raw_data = dataset['raw_data']
    windows = dataset['windowed_data']
    print(f"  Total samples: {len(raw_data)}")
    print(f"  Windows: {len(windows)}")

    # --- Preprocess ---
    print("\n[2/4] Preprocessing...")
    # Gas label columns in raw_data are: 'h', 'n', 'c', 'e'
    # Get baseline (h=0, n=0, c=0, e=0 means all gases absent)
    baseline_mask = (raw_data['h'] == 0) & (raw_data['n'] == 0) & \
                    (raw_data['c'] == 0) & (raw_data['e'] == 0)
    if baseline_mask.sum() > 0:
        baseline = raw_data.loc[baseline_mask, SENSOR_COLS].mean()
    else:
        baseline = None
        print("  Warning: No baseline (all-absent) data found")

    # Preprocess each window and extract labels
    processed_windows = []
    labels = []
    for w in windows:
        sensor_data = w['window']  # numpy array (window_size, 6)
        df_w = pd.DataFrame(sensor_data, columns=SENSOR_COLS)

        # Baseline correction
        if baseline is not None:
            df_w = df_w - baseline

        # EWMA smoothing
        df_w = df_w.ewm(alpha=0.1, adjust=False).mean()
        processed_windows.append(df_w)

        # Extract labels from window dict
        lbl = w['labels']
        labels.append([lbl['h'], lbl['n'], lbl['c'], lbl['e']])

    labels = np.array(labels)
    print(f"  Processed {len(processed_windows)} windows")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Class distribution: H2S={labels[:,0].sum()}, NH3={labels[:,1].sum()}, "
          f"CO={labels[:,2].sum()}, C2H5OH={labels[:,3].sum()}")

    # --- Feature Extraction ---
    print("\n[3/4] Extracting features (126 features per sample)...")
    t0 = time.time()
    feature_list = []

    # Generate feature names
    sensor_short = ['402b', '602b', '502b', '702b', 'tgs8100', '802b']
    stat_names = ['mean', 'median', 'std', 'var', 'min', 'max', 'range', 'skew', 'kurtosis']
    trans_names = ['rise_time', 'slope', 'integral', 'peak', 'time_to_peak', 'settle_time']
    freq_names = ['dom_freq', 'spectral_energy', 'spectral_centroid']
    deriv_names = ['max_dx', 'mean_dx', 'zero_cross_rate']

    feature_names = []
    for feat_group, group_names in [(stat_names, 'stat'), (trans_names, 'trans'),
                                     (freq_names, 'freq'), (deriv_names, 'deriv')]:
        for feat in feat_group if group_names == 'stat' else feat_group:
            for s in sensor_short:
                feature_names.append(f'{s}_{feat}')

    # Reorder: stat features are (9 features × 6 sensors stacked), others are per-sensor
    feature_names = []
    for feat in stat_names:
        for s in sensor_short:
            feature_names.append(f'{s}_{feat}')
    for feat in trans_names:
        for s in sensor_short:
            feature_names.append(f'{s}_{feat}')
    for feat in freq_names:
        for s in sensor_short:
            feature_names.append(f'{s}_{feat}')
    for feat in deriv_names:
        for s in sensor_short:
            feature_names.append(f'{s}_{feat}')

    for i, w in enumerate(processed_windows):
        feats = extract_all_features(w.values)
        feature_list.append(feats)
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(processed_windows)} windows processed...")

    X = np.array(feature_list)
    y = labels
    print(f"  Features extracted in {time.time()-t0:.1f}s")
    print(f"  Feature matrix: {X.shape}")

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Save ---
    print("\n[4/4] Saving processed data...")
    feat_dir = DATA_DIR / 'features'
    feat_dir.mkdir(parents=True, exist_ok=True)
    np.save(feat_dir / 'X_features.npy', X)
    np.save(feat_dir / 'y_labels.npy', y)
    with open(feat_dir / 'feature_names.json', 'w') as f:
        json.dump(feature_names if feature_names else [], f)
    print(f"  Saved to {feat_dir}")

    return X, y, feature_names


def phase_explore(X, y, feature_names):
    """Phase 3: Exploratory analysis."""
    print("\n" + "=" * 70)
    print("  PHASE 3: EXPLORATORY ANALYSIS")
    print("=" * 70)

    from exploration.analysis import (
        plot_pca_2d, plot_pca_3d, plot_tsne, plot_lda,
        plot_correlation_heatmap, plot_class_distribution,
        plot_kmeans_clustering
    )

    fig_dir = str(RESULTS_DIR / 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # Create composite labels for plotting
    gas_names = ['H₂S', 'NH₃', 'CO', 'C₂H₅OH']
    y_labels = []
    for row in y:
        present = [gas_names[i] for i in range(4) if row[i] == 1]
        y_labels.append('+'.join(present) if present else 'None')
    y_labels = np.array(y_labels)

    sensor_names = ['402b', '602b', '502b', '702b', 'TGS8100', '802b']

    print("[1/6] PCA 2D...")
    plot_pca_2d(X, y_labels, fig_dir)

    print("[2/6] PCA 3D...")
    plot_pca_3d(X, y_labels, fig_dir)

    print("[3/6] t-SNE...")
    # Use subset for t-SNE (slow on full data)
    n_tsne = min(2000, len(X))
    idx = np.random.choice(len(X), n_tsne, replace=False)
    plot_tsne(X[idx], y_labels[idx], fig_dir)

    print("[4/6] LDA...")
    plot_lda(X, y_labels, fig_dir)

    print("[5/6] Correlation heatmap...")
    plot_correlation_heatmap(X[:, :6], sensor_names, fig_dir)

    print("[6/6] Class distribution...")
    plot_class_distribution(y_labels, gas_names, fig_dir)

    print(f"\n  All exploration plots saved to {fig_dir}")


def phase_classify(X, y):
    """Phase 4: Classification with all models."""
    print("\n" + "=" * 70)
    print("  PHASE 4: CLASSIFICATION (13 MODELS)")
    print("=" * 70)

    from models.traditional_ml import run_all_traditional_models
    from evaluation.metrics import (
        compute_classification_metrics, plot_confusion_matrices,
        plot_model_comparison_bar, plot_radar_chart, plot_roc_curves,
        generate_latex_table
    )

    fig_dir = str(RESULTS_DIR / 'figures')
    tab_dir = str(RESULTS_DIR / 'tables')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    all_results = {}
    all_preds = {}
    all_probs = {}
    timings = {}

    # --- Tier 1: Traditional ML ---
    print("\n--- Tier 1: Traditional ML ---")
    t0 = time.time()
    try:
        trad_results = run_all_traditional_models(X_train, X_test, y_train, y_test, task='classification')

        for name, res in trad_results.items():
            if name == 'KMeans':
                print(f"  KMeans clustering: {res}")
                continue
            if 'predictions' not in res:
                continue

            y_pred = np.array(res['predictions'])
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
            if y_pred.shape[1] != y_test.shape[1]:
                print(f"  Skipping {name}: shape mismatch {y_pred.shape} vs {y_test.shape}")
                continue

            # Get probabilities if available
            y_prob = None
            model = res.get('model')
            if model is not None and hasattr(model, 'predict_proba'):
                try:
                    y_prob = model.predict_proba(X_test)
                    if isinstance(y_prob, list):
                        y_prob = np.column_stack([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in y_prob])
                except Exception:
                    pass

            metrics = compute_classification_metrics(y_test, y_pred, y_prob)
            all_results[name] = metrics
            all_preds[name] = y_pred
            if y_prob is not None:
                all_probs[name] = y_prob
            elapsed = time.time() - t0
            timings[name] = elapsed
            print(f"  {name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")

    except Exception as e:
        print(f"  Traditional ML failed: {e}")
        import traceback
        traceback.print_exc()

    # --- Tier 2 & 3: Neural Networks ---
    print("\n--- Tier 2 & 3: Neural Networks ---")
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Device: {device}")

        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)
        X_test_t = torch.FloatTensor(X_test)
        y_test_t = torch.FloatTensor(y_test)

        train_ds = TensorDataset(X_train_t, y_train_t)
        test_ds = TensorDataset(X_test_t, y_test_t)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        from models.neural_networks import ANN, CNN1D, LSTM_Attention, TCN_MHA
        from models.advanced_models import TransformerENose, GNN, PIML_Model, PIML_Loss

        n_features = X_train.shape[1]
        n_outputs = y_train.shape[1]

        nn_models = {
            'ANN': ANN(n_features, n_outputs, task='classification'),
            '1D-CNN': CNN1D(6, n_outputs, task='classification'),
            'LSTM': LSTM_Attention(input_dim=6, hidden_dim=64, output_dim=n_outputs, task='classification'),
            'TCN': TCN_MHA(in_channels=6, output_dim=n_outputs, task='classification'),
            'Transformer': TransformerENose(in_channels=6, d_model=64, nhead=4, output_dim=n_outputs, task='classification'),
            'GNN': GNN(in_features=21, hidden_features=64, output_dim=n_outputs, num_nodes=6, task='classification'),
            'PIML': PIML_Model(n_features, n_outputs, task='classification'),
        }

        for name, model in nn_models.items():
            print(f"\n  Training {name}...")
            t0 = time.time()
            try:
                model = model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
                criterion = PIML_Loss(task='classification') if name == 'PIML' else torch.nn.BCEWithLogitsLoss()
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

                best_loss = float('inf')
                patience_counter = 0
                for epoch in range(40):
                    model.train()
                    epoch_loss = 0
                    for xb, yb in train_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        optimizer.zero_grad()
                        out = model(xb)
                        if name == 'PIML':
                            loss = criterion(out, yb, xb)
                        else:
                            loss = criterion(out, yb)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()

                    avg_loss = epoch_loss / len(train_loader)
                    scheduler.step(avg_loss)
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    if patience_counter >= 8:
                        break

                # Evaluate
                model.eval()
                all_out = []
                with torch.no_grad():
                    for xb, yb in test_loader:
                        xb = xb.to(device)
                        out = model(xb)
                        all_out.append(torch.sigmoid(out).cpu().numpy())

                y_prob = np.concatenate(all_out, axis=0)
                y_pred = (y_prob >= 0.5).astype(int)
                elapsed = time.time() - t0
                timings[name] = elapsed

                metrics = compute_classification_metrics(y_test, y_pred, y_prob)
                all_results[name] = metrics
                all_preds[name] = y_pred
                all_probs[name] = y_prob

                print(f"    {name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f} ({elapsed:.1f}s)")

            except Exception as e:
                print(f"    Failed: {name}: {e}")
                import traceback
                traceback.print_exc()

    except ImportError as e:
        print(f"  PyTorch not available, skipping neural networks: {e}")

    # --- Generate Results ---
    print("\n\n--- Generating IEEE Classification Results ---")

    # Confusion matrices
    for name, y_pred in all_preds.items():
        plot_confusion_matrices(y_test, y_pred, name, fig_dir)

    # ROC curves
    if all_probs:
        plot_roc_curves(y_test, all_probs, fig_dir)

    # Comparison bar chart
    if all_results:
        clf_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        plot_model_comparison_bar(all_results, clf_metrics, fig_dir, 'classification_comparison')

        # Radar chart
        radar_metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        plot_radar_chart(all_results, radar_metrics, fig_dir)

        # LaTeX table
        generate_latex_table(
            all_results,
            ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'hamming_loss', 'mcc_mean'],
            'Classification Performance Comparison of All Models',
            'tab:classification_all',
            os.path.join(tab_dir, 'classification_all_models.tex')
        )

        # Per-gas tables
        for gas in ['H₂S', 'NH₃', 'CO', 'C₂H₅OH']:
            gas_metrics = [f'{gas}_precision', f'{gas}_recall', f'{gas}_f1']
            generate_latex_table(
                all_results, gas_metrics,
                f'Per-Gas Classification for {gas}',
                f'tab:{gas}_detail',
                os.path.join(tab_dir, f'{gas.lower().replace("₂","2").replace("₅","5")}_detail.tex')
            )

        # Timing table
        timing_results = {m: {'train_time_s': timings.get(m, 0)} for m in all_results}
        generate_latex_table(
            timing_results, ['train_time_s'],
            'Computational Cost Comparison',
            'tab:timing',
            os.path.join(tab_dir, 'computational_cost.tex')
        )

    # Summary
    print("\n" + "=" * 70)
    print("  CLASSIFICATION RESULTS SUMMARY")
    print("=" * 70)
    summary_df = pd.DataFrame(all_results).T
    summary_cols = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    available_cols = [c for c in summary_cols if c in summary_df.columns]
    if available_cols:
        print(summary_df[available_cols].round(4).to_string())

    return all_results, all_preds, all_probs, X_test, y_test


def phase_regression():
    """Phase 5: Concentration regression on UCI benchmark datasets."""
    print("\n" + "=" * 70)
    print("  PHASE 5: CONCENTRATION REGRESSION (UCI DATASETS)")
    print("=" * 70)

    from preprocessing.load_uci_data import load_uci_487_co_regression
    from evaluation.metrics import (
        compute_regression_metrics, plot_predicted_vs_actual, generate_latex_table
    )
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor

    fig_dir = str(RESULTS_DIR / 'figures')
    tab_dir = str(RESULTS_DIR / 'tables')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    print("\n[1/3] Loading UCI 487 (CO Continuous Concentration 0-20 ppm)...")
    X, y, feature_names = load_uci_487_co_regression()
    print(f"  Samples: {len(X)}, Features: {X.shape[1]}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    reg_results = {}
    reg_preds = {}

    # --- Traditional Models ---
    print("\n[2/3] Training Regression Models...")
    models = {
        'KNN Regressor': KNeighborsRegressor(n_neighbors=5, weights='distance'),
        'Support Vector Regressor': SVR(C=10.0, gamma='scale'),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        'XGBoost Regressor': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1),
    }

    for name, model in models.items():
        t0 = time.time()
        print(f"  Training {name}...")
        model.fit(X_train, y_train.ravel())
        y_pred = model.predict(X_test).reshape(-1, 1)
        elapsed = time.time() - t0
        metrics = compute_regression_metrics(y_test, y_pred, ['CO'])
        reg_results[name] = metrics
        reg_preds[name] = y_pred
        print(f"    R²={metrics['CO_r2']:.4f}, RMSE={metrics['CO_rmse']:.4f} ppm, MAE={metrics['CO_mae']:.4f} ppm ({elapsed:.1f}s)")

    # --- Deep Learning Regressors ---
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from models.neural_networks import ANN, CNN1D, TCN_MHA

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        dl_reg_models = {
            'ANN Regressor': ANN(X_train.shape[1], 1, task='regression'),
            '1D-CNN Regressor': CNN1D(in_channels=1, output_dim=1, task='regression'),
            'TCN Regressor': TCN_MHA(in_channels=1, output_dim=1, task='regression')
        }

        for name, model in dl_reg_models.items():
            t0 = time.time()
            print(f"  Training {name}...")
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
            criterion = torch.nn.MSELoss()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

            for epoch in range(35):
                model.train()
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    out = model(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()

            model.eval()
            preds_dl = []
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb = xb.to(device)
                    preds_dl.append(model(xb).cpu().numpy())
            y_pred = np.concatenate(preds_dl, axis=0)
            elapsed = time.time() - t0
            metrics = compute_regression_metrics(y_test, y_pred, ['CO'])
            reg_results[name] = metrics
            reg_preds[name] = y_pred
            print(f"    R²={metrics['CO_r2']:.4f}, RMSE={metrics['CO_rmse']:.4f} ppm, MAE={metrics['CO_mae']:.4f} ppm ({elapsed:.1f}s)")

    except Exception as e:
        print(f"  Deep learning regression skipped: {e}")

    # --- Generate IEEE Regression Results ---
    print("\n[3/3] Generating IEEE Regression Results...")
    for name, y_pred in reg_preds.items():
        plot_predicted_vs_actual(y_test, y_pred, ['CO'], name, fig_dir)

    reg_metrics_keys = ['CO_r2', 'CO_rmse', 'CO_mae', 'CO_mape']
    generate_latex_table(
        reg_results,
        reg_metrics_keys,
        'CO Gas Concentration Prediction on UCI Benchmark Dataset',
        'tab:uci_regression',
        os.path.join(tab_dir, 'uci_co_regression.tex')
    )

    print("\n" + "=" * 70)
    print("  REGRESSION RESULTS SUMMARY (UCI 487 CO 0-20 ppm)")
    print("=" * 70)
    summary_df = pd.DataFrame(reg_results).T
    print(summary_df[['CO_r2', 'CO_rmse', 'CO_mae']].round(4).to_string())

    return reg_results, reg_preds


def main():
    parser = argparse.ArgumentParser(description='E-Nose Research Pipeline')
    parser.add_argument('--phase', type=str, default='all',
                        choices=['all', 'preprocess', 'explore', 'classify', 'regression'],
                        help='Pipeline phase to run')
    parser.add_argument('--skip-uci', action='store_true', help='Skip UCI datasets')
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  E-NOSE RESEARCH PIPELINE")
    print("  Gases: NH₃, H₂S, CO, C₂H₅OH")
    print("  Models: KNN, SVM, RF, XGBoost, LDA, ANN, CNN, LSTM, TCN,")
    print("          Transformer, GNN, Physics-Informed ML")
    print("=" * 70)

    total_start = time.time()

    # Phase 1 & 2: Load + Preprocess + Features
    feat_dir = DATA_DIR / 'features'
    if (feat_dir / 'X_features.npy').exists() and args.phase != 'preprocess':
        print("\n  Loading cached features...")
        X = np.load(feat_dir / 'X_features.npy')
        y = np.load(feat_dir / 'y_labels.npy')
        if (feat_dir / 'feature_names.json').exists():
            with open(feat_dir / 'feature_names.json', 'r', encoding='utf-8') as f:
                feature_names = json.load(f)
        else:
            feature_names = [f'feat_{i}' for i in range(X.shape[1])]
        print(f"  Loaded: X={X.shape}, y={y.shape}")
    else:
        X, y, feature_names = phase_preprocess()

    if args.phase in ['all', 'explore']:
        phase_explore(X, y, feature_names)

    if args.phase in ['all', 'classify']:
        results_clf = phase_classify(X, y)

    if args.phase in ['all', 'regression'] and not args.skip_uci:
        results_reg = phase_regression()

    total_time = time.time() - total_start
    print(f"\n\n  Total pipeline time: {total_time/60:.1f} minutes")
    print(f"  Results saved to: {RESULTS_DIR}")
    print("\n  Done! 🚀\n")


if __name__ == '__main__':
    main()
