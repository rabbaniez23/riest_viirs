"""
STEP 3: predict_xgboost.py
----------------------------
Model XGBoost untuk prediksi sebaran spasial deteksi kapal.
Pendekatan: Grid 0.5°×0.5° per bulan sebagai unit analisis.
Target: Jumlah deteksi per grid per bulan.
Output: Peta prediksi density 2026-2030 + evaluasi model.

Jalankan: python predict_xgboost.py
Pastikan process_viirs_wpp.py sudah dijalankan terlebih dahulu!
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon as MplPolygon
import shapefile
from shapely.geometry import shape
from shapely.ops import unary_union
import re
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import joblib
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False
    print("[WARNING] xgboost/scikit-learn/joblib belum terinstall.")
    print("Jalankan: pip install xgboost scikit-learn joblib")

# ─── Konfigurasi Path ─────────────────────────────────────────────────────────
FILTERED_CSV = 'd:/Pekerjaan/riset/pertemuan4/output/filtered_data.csv'
SHP_DIR      = 'd:/Pekerjaan/riset/pertemuan4/shp'
PRED_DIR     = 'd:/Pekerjaan/riset/pertemuan4/output/predictions'
FIG_DIR      = 'd:/Pekerjaan/riset/pertemuan4/output/figures'
MODEL_PATH   = 'd:/Pekerjaan/riset/pertemuan4/output/predictions/xgboost_model.pkl'

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Grid resolusi
GRID_RES   = 0.5   # derajat
LON_MIN, LON_MAX = 90.0, 145.0
LAT_MIN, LAT_MAX = -12.0, 12.0
PREDICT_YEARS = list(range(2026, 2031))  # 2026-2030

SAVE_DPI = 300

plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.titlesize': 11,
                     'axes.labelsize': 10, 'figure.dpi': 150})


# ──────────────────────────────────────────────────────────────────────────────
def load_wpp_geoms():
    wpp_geoms = {}
    for fn in sorted(os.listdir(SHP_DIR)):
        if not fn.endswith('.shp'):
            continue
        m = re.search(r'WPP-RI\s+(\d+)', fn)
        code = m.group(1) if m else fn.replace('.shp', '')
        fp = os.path.join(SHP_DIR, fn)
        try:
            sf = shapefile.Reader(shp=fp)
            shapes = [shape(s) for s in sf.shapes() if s.shapeType != 0]
            if shapes:
                wpp_geoms[code] = shapes[0] if len(shapes) == 1 else unary_union(shapes)
        except:
            pass
    return wpp_geoms


def draw_wpp_boundaries(ax, wpp_geoms):
    WPP_CODES = ['571','572','573','711','712','713','714','715','716','717','718']
    CMAP_TAB  = plt.get_cmap('tab20')
    wpp_color = {c: CMAP_TAB(i/len(WPP_CODES)) for i, c in enumerate(WPP_CODES)}
    for code, geom in wpp_geoms.items():
        polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
        for poly in polys:
            x, y = poly.exterior.xy
            ax.plot(x, y, color='#333333', linewidth=0.5, zorder=3)
            patch = MplPolygon(np.column_stack((x, y)),
                               facecolor=wpp_color.get(code, 'gray'),
                               alpha=0.12, edgecolor='none', zorder=2)
            ax.add_patch(patch)
        cx, cy = geom.centroid.x, geom.centroid.y
        ax.text(cx, cy, code, fontsize=7, ha='center', va='center',
                color='#111111', fontweight='bold', zorder=5)


# ──────────────────────────────────────────────────────────────────────────────
# Buat fitur grid spasial-temporal
# ──────────────────────────────────────────────────────────────────────────────
def make_grid_features(df):
    """
    Konversi koordinat kapal ke grid 0.5° dan agregat per (year, month, lat_grid, lon_grid).
    Kembalikan DataFrame fitur untuk XGBoost.
    """
    df2 = df.copy()
    df2['lat_grid'] = (np.floor(df2['Lat_DNB'] / GRID_RES) * GRID_RES).round(2)
    df2['lon_grid'] = (np.floor(df2['Lon_DNB'] / GRID_RES) * GRID_RES).round(2)

    agg = df2.groupby(['Year', 'Month', 'lat_grid', 'lon_grid']).agg(
        count=('Lat_DNB', 'count'),
        mean_rad=('Rad_DNB', 'mean')
    ).reset_index()

    # Tambah fitur waktu
    agg['month_sin'] = np.sin(2 * np.pi * agg['Month'] / 12)
    agg['month_cos'] = np.cos(2 * np.pi * agg['Month'] / 12)
    agg['year_norm'] = (agg['Year'] - agg['Year'].min()) / max(agg['Year'].max() - agg['Year'].min(), 1)
    agg['time_idx']  = (agg['Year'] - agg['Year'].min()) * 12 + agg['Month']

    # Lag fitur (t-1 bulan)
    agg = agg.sort_values(['lat_grid', 'lon_grid', 'Year', 'Month'])
    agg['lag1'] = agg.groupby(['lat_grid', 'lon_grid'])['count'].shift(1).fillna(0)
    agg['lag2'] = agg.groupby(['lat_grid', 'lon_grid'])['count'].shift(2).fillna(0)
    agg['roll3'] = agg.groupby(['lat_grid', 'lon_grid'])['count'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0)

    return agg


def make_future_grid(lat_range, lon_range, years, months=range(1, 13)):
    """Buat grid kosong untuk tahun masa depan."""
    rows = []
    for yr in years:
        for mo in months:
            for lat in lat_range:
                for lon in lon_range:
                    rows.append({'Year': yr, 'Month': mo,
                                 'lat_grid': lat, 'lon_grid': lon})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN & PREDICT
# ──────────────────────────────────────────────────────────────────────────────
def train_xgboost(agg):
    FEATURE_COLS = ['Year', 'Month', 'lat_grid', 'lon_grid',
                    'month_sin', 'month_cos', 'year_norm',
                    'time_idx', 'lag1', 'lag2', 'roll3', 'mean_rad']
    TARGET = 'count'

    # Pastikan semua fitur ada
    feature_cols = [c for c in FEATURE_COLS if c in agg.columns]
    X = agg[feature_cols].fillna(0).values
    y = agg[TARGET].values

    print(f"\nTraining XGBoost...")
    print(f"  Jumlah sampel (grid-bulan): {len(X)}")
    print(f"  Fitur: {feature_cols}")

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0
    )

    if len(X) >= 5:
        # Time-series cross validation (tidak shuffle)
        cv_folds = min(5, len(X) // 2)
        scores_r2  = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
        scores_mae = cross_val_score(model, X, y, cv=cv_folds,
                                     scoring='neg_mean_absolute_error')
        print(f"\n  Cross-Validation ({cv_folds} fold):")
        print(f"    R²  : {scores_r2.mean():.4f} ± {scores_r2.std():.4f}")
        print(f"    MAE : {-scores_mae.mean():.4f} ± {scores_mae.std():.4f}")

    model.fit(X, y)

    # Evaluasi pada data training
    y_pred = model.predict(X)
    print(f"\n  Evaluasi pada data training:")
    print(f"    R²   : {r2_score(y, y_pred):.4f}")
    print(f"    MAE  : {mean_absolute_error(y, y_pred):.4f}")
    print(f"    RMSE : {np.sqrt(mean_squared_error(y, y_pred)):.4f}")

    # Simpan model
    if 'joblib' in dir():
        joblib.dump(model, MODEL_PATH)
        print(f"\n  Model tersimpan: {MODEL_PATH}")

    return model, feature_cols


def predict_future(model, feature_cols, lat_range, lon_range):
    """Prediksi kepadatan kapal 2026-2030 per grid per bulan/tahun."""
    future_df = make_future_grid(lat_range, lon_range, PREDICT_YEARS)
    future_df['month_sin'] = np.sin(2 * np.pi * future_df['Month'] / 12)
    future_df['month_cos'] = np.cos(2 * np.pi * future_df['Month'] / 12)
    future_df['year_norm'] = (future_df['Year'] - 2024) / 10
    future_df['time_idx']  = (future_df['Year'] - 2024) * 12 + future_df['Month']
    future_df['lag1']      = 0
    future_df['lag2']      = 0
    future_df['roll3']     = 0
    future_df['mean_rad']  = 5.0  # asumsi rata-rata historis

    cols = [c for c in feature_cols if c in future_df.columns]
    X_future = future_df[cols].fillna(0).values
    future_df['predicted_count'] = np.maximum(0, model.predict(X_future))
    return future_df


# ──────────────────────────────────────────────────────────────────────────────
# VISUALISASI PREDIKSI
# ──────────────────────────────────────────────────────────────────────────────
def plot_prediction_maps(pred_df, wpp_geoms):
    """Buat peta prediksi annual (rata-rata per tahun)."""
    print("\nMembuat peta prediksi 2026-2030...")
    annual_pred = pred_df.groupby(['Year', 'lat_grid', 'lon_grid'])['predicted_count'].sum().reset_index()

    lon_bins = np.arange(LON_MIN, LON_MAX + GRID_RES, GRID_RES)
    lat_bins = np.arange(LAT_MIN, LAT_MAX + GRID_RES, GRID_RES)

    n_years = len(PREDICT_YEARS)
    fig, axes = plt.subplots(1, n_years, figsize=(5 * n_years, 6), sharey=True)
    if n_years == 1:
        axes = [axes]

    vmax = annual_pred['predicted_count'].quantile(0.98)
    vmax = max(vmax, 1)

    for ax, yr in zip(axes, PREDICT_YEARS):
        sub = annual_pred[annual_pred['Year'] == yr]
        H = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1))
        for _, row in sub.iterrows():
            li = np.searchsorted(lat_bins, row['lat_grid']) - 1
            lo = np.searchsorted(lon_bins, row['lon_grid']) - 1
            if 0 <= li < H.shape[0] and 0 <= lo < H.shape[1]:
                H[li, lo] = row['predicted_count']
        H_m = np.ma.masked_where(H == 0, H)

        ax.set_facecolor('#A8D5E8')
        im = ax.pcolormesh(lon_bins, lat_bins, H_m,
                           cmap='YlOrRd',
                           norm=mcolors.Normalize(vmin=0, vmax=vmax),
                           alpha=0.85, zorder=6)
        draw_wpp_boundaries(ax, wpp_geoms)
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.set_title(f'Prediksi {yr}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Bujur', fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel('Lintang', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3, zorder=1)

    plt.colorbar(im, ax=axes, label='Prediksi Deteksi Kapal (akumulasi tahunan)', shrink=0.7)
    fig.suptitle('Prediksi Sebaran Spasial Kapal VIIRS 2026–2030\n(Model XGBoost — Grid 0.5°×0.5°)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig7_xgboost_spatial_prediction.png')
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Tersimpan: {out}")


def plot_wpp_prediction_trend(pred_df, wpp_geoms):
    """Plot prediksi total deteksi per WPP per tahun."""
    from shapely.prepared import prep
    from shapely.geometry import Point

    print("\nMenghitung prediksi per WPP-RI...")
    annual = pred_df.groupby(['Year', 'lat_grid', 'lon_grid'])['predicted_count'].sum().reset_index()
    prepared = {code: prep(geom) for code, geom in wpp_geoms.items()}

    # Assign WPP ke setiap grid
    wpp_labels = []
    for _, row in annual.iterrows():
        p = Point(row['lon_grid'] + GRID_RES/2, row['lat_grid'] + GRID_RES/2)
        w = 'Outside'
        for code, pgeom in prepared.items():
            if pgeom.contains(p):
                w = code
                break
        wpp_labels.append(w)
    annual['WPP_RI'] = wpp_labels

    wpp_yr = annual[annual['WPP_RI'] != 'Outside'].groupby(
        ['Year', 'WPP_RI'])['predicted_count'].sum().reset_index()

    WPP_CODES = sorted(wpp_yr['WPP_RI'].unique())
    if not WPP_CODES:
        print("  Tidak ada grid yang masuk WPP-RI, skip.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    CMAP_TAB = plt.get_cmap('tab20')
    for i, code in enumerate(WPP_CODES):
        sub = wpp_yr[wpp_yr['WPP_RI'] == code].sort_values('Year')
        ax.plot(sub['Year'], sub['predicted_count'], 'o-',
                color=CMAP_TAB(i/len(WPP_CODES)), linewidth=1.8, markersize=7, label=f'WPP-RI {code}')

    ax.set_xlabel('Tahun Prediksi', fontsize=11)
    ax.set_ylabel('Prediksi Jumlah Deteksi Kapal', fontsize=11)
    ax.set_title('Tren Prediksi Jumlah Deteksi Kapal per WPP-RI (2026–2030)\nModel: XGBoost',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9)
    ax.set_xticks(PREDICT_YEARS)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig8_xgboost_wpp_trend.png')
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Tersimpan: {out}")

    # Simpan ringkasan prediksi ke CSV
    out_csv = os.path.join(PRED_DIR, 'prediction_summary.csv')
    wpp_yr.to_csv(out_csv, index=False)
    print(f"  Ringkasan tersimpan: {out_csv}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    if not XGBOOST_OK:
        print("Hentikan: xgboost tidak tersedia. Jalankan: pip install xgboost scikit-learn joblib")
        return

    if not os.path.exists(FILTERED_CSV):
        print(f"ERROR: {FILTERED_CSV} tidak ditemukan.")
        print("Jalankan dulu: python process_viirs_wpp.py")
        return

    print(f"Memuat data: {FILTERED_CSV}")
    df = pd.read_csv(FILTERED_CSV, parse_dates=['Date'])
    print(f"Total data: {len(df)} baris")
    print(f"Rentang: {df['Date'].min()} s/d {df['Date'].max()}")

    if len(df) < 20:
        print("\n[WARNING] Data terlalu sedikit untuk prediksi yang andal.")
        print("Tambahkan lebih banyak file VIIRS untuk hasil yang lebih baik.")

    # Buat fitur grid
    agg = make_grid_features(df)
    print(f"\nGrid features: {len(agg)} sampel (kombinasi grid × bulan)")

    # Train model
    model, feature_cols = train_xgboost(agg)

    # Define range grid
    lat_range = np.arange(LAT_MIN, LAT_MAX, GRID_RES)
    lon_range = np.arange(LON_MIN, LON_MAX, GRID_RES)

    # Prediksi
    pred_df = predict_future(model, feature_cols, lat_range, lon_range)
    print(f"\nPrediksi dibuat: {len(pred_df)} baris (grid × bulan × tahun)")

    # Load WPP geoms
    wpp_geoms = load_wpp_geoms()

    # Visualisasi
    plot_prediction_maps(pred_df, wpp_geoms)
    plot_wpp_prediction_trend(pred_df, wpp_geoms)

    print("\n[SELESAI] Pipeline prediksi XGBoost selesai!")
    print(f"Semua output tersimpan di:")
    print(f"  Gambar  : output/figures/")
    print(f"  Prediksi: output/predictions/")

if __name__ == '__main__':
    main()
