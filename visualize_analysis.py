"""
STEP 2: visualize_analysis.py
------------------------------
Membuat 6 visualisasi akademis dari output/filtered_data.csv:
  Fig 1: Peta sebaran kapal di atas basemap Indonesia (dari Natural Earth)
  Fig 2: Annual count per WPP-RI + regresi linear
  Fig 3: Pola bulanan per tahun (multi-line chart)
  Fig 4: Histogram distribusi intensitas Rad_DNB per WPP-RI
  Fig 5: Density heatmap spasial (grid 0.5°x0.5°)
  Fig 6: Latitude-Time distribution heatmap

Jalankan: python visualize_analysis.py
Pastikan process_viirs_wpp.py sudah dijalankan terlebih dahulu!
"""

import os
import requests
import zipfile
import io
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon as MplPolygon
from scipy import stats
# pyrefly: ignore [missing-import]
import shapefile
from shapely.geometry import shape
from shapely.ops import unary_union

warnings.filterwarnings('ignore')

# ─── Konfigurasi Path (Relative) ──────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FILTERED_CSV = os.path.join(BASE_DIR, 'output', 'filtered_data.csv')
SHP_DIR      = os.path.join(BASE_DIR, 'shp&shx')
FIG_DIR      = os.path.join(BASE_DIR, 'output', 'figures')

# ─── Style Global ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 150
})
SAVE_DPI = 300

# ─── Warna WPP ────────────────────────────────────────────────────────────────
WPP_CODES = ['571', '572', '573', '711', '712', '713', '714', '715', '716', '717', '718']
CMAP_TAB  = plt.get_cmap('tab20')
WPP_COLOR = {c: CMAP_TAB(i / len(WPP_CODES)) for i, c in enumerate(WPP_CODES)}

os.makedirs(FIG_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: Unduh basemap Indonesia dari Natural Earth
# ══════════════════════════════════════════════════════════════════════════════
def get_indonesia_basemap():
    extract_dir = os.path.join(BASE_DIR, 'output', 'ne_countries')
    if not os.path.exists(extract_dir):
        print("Mengunduh basemap Indonesia dari Natural Earth...")
        url = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
        try:
            r = requests.get(url, timeout=60)
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                z.extractall(extract_dir)
            print("  Download selesai.")
        except Exception as e:
            print(f"  Gagal unduh basemap: {e}. Peta tanpa daratan.")
            return []

    shp_files = [f for f in os.listdir(extract_dir) if f.endswith('.shp')]
    if not shp_files:
        return []

    shp_path = os.path.join(extract_dir, shp_files[0])
    # Gunakan path tanpa ekstensi agar pyshp otomatis membaca .dbf dan .shx
    base_path = shp_path.replace('.shp', '')
    sf = shapefile.Reader(base_path)
    fields = [f[0] for f in sf.fields[1:]]
    indonesia_polys = []
    for rec, sh in zip(sf.records(), sf.shapes()):
        rec_dict = dict(zip(fields, rec))
        name = rec_dict.get('ADMIN', '') or rec_dict.get('NAME', '') or ''
        if 'Indonesia' in str(name):
            indonesia_polys.append(shape(sh))
    return indonesia_polys


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: Load WPP Geometries
# ══════════════════════════════════════════════════════════════════════════════
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
    for code, geom in wpp_geoms.items():
        geom = geom.simplify(0.05, preserve_topology=True)
        color = WPP_COLOR.get(code, 'gray')
        polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
        for poly in polys:
            x, y = poly.exterior.xy
            ax.plot(x, y, color='#333333', linewidth=0.6, zorder=3)
            patch = MplPolygon(np.column_stack((x, y)),
                               facecolor=color, alpha=0.18, edgecolor='none', zorder=2)
            ax.add_patch(patch)
        cx, cy = geom.centroid.x, geom.centroid.y
        ax.text(cx, cy, code, fontsize=7, ha='center', va='center',
                color='#111111', fontweight='bold', zorder=5)


def draw_indonesia_land(ax, indonesia_polys):
    for geom in indonesia_polys:
        geom = geom.simplify(0.05, preserve_topology=True)
        polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
        for poly in polys:
            x, y = poly.exterior.xy
            patch = MplPolygon(np.column_stack((x, y)),
                               facecolor='#C8D6AF', edgecolor='#888888',
                               linewidth=0.4, zorder=4)
            ax.add_patch(patch)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1: PETA SEBARAN KAPAL + BASEMAP INDONESIA
# ══════════════════════════════════════════════════════════════════════════════
def fig1_map(df, wpp_geoms, indonesia_polys):
    print("Membuat Fig 1: Peta Sebaran Kapal...")
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_facecolor('#A8D5E8')

    draw_indonesia_land(ax, indonesia_polys)
    draw_wpp_boundaries(ax, wpp_geoms)

    df_in  = df[df['WPP_RI'] != 'Outside']
    df_out = df[df['WPP_RI'] == 'Outside']

    if not df_out.empty:
        ax.scatter(df_out['Lon_DNB'], df_out['Lat_DNB'],
                   c='#AAAAAA', s=4, alpha=0.4, label='Luar WPP-RI', zorder=6)

    for code in WPP_CODES:
        sub = df_in[df_in['WPP_RI'] == code]
        if not sub.empty:
            ax.scatter(sub['Lon_DNB'], sub['Lat_DNB'],
                       c=[WPP_COLOR[code]], s=6, alpha=0.75,
                       label=f'WPP-RI {code} (n={len(sub)})', zorder=7)

    all_bounds = [g.bounds for g in wpp_geoms.values()]
    ax.set_xlim(min(b[0] for b in all_bounds) - 1, max(b[2] for b in all_bounds) + 1)
    ax.set_ylim(min(b[1] for b in all_bounds) - 1, max(b[3] for b in all_bounds) + 1)
    ax.set_xlabel('Bujur (Longitude)', fontsize=11)
    ax.set_ylabel('Lintang (Latitude)', fontsize=11)
    ax.set_title(
        f'Sebaran Deteksi Kapal VIIRS di Wilayah WPP-RI Indonesia\n'
        f'(Total: {len(df)} titik | QF_Detect=1 | {df["Date"].min()} s/d {df["Date"].max()})',
        fontsize=13, fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.4, zorder=1)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8, framealpha=0.9)

    out = os.path.join(FIG_DIR, 'fig1_map_wpp_boat_detections.png')
    plt.tight_layout()
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Tersimpan: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2: ANNUAL COUNT + REGRESI LINEAR
# ══════════════════════════════════════════════════════════════════════════════
def fig2_annual_trend(df):
    print("Membuat Fig 2: Tren Tahunan per WPP-RI...")
    wpp_in = [c for c in WPP_CODES if c in df['WPP_RI'].unique()]
    if not wpp_in:
        print("  Tidak ada data dalam WPP-RI, skip.")
        return

    annual = (df[df['WPP_RI'].isin(wpp_in)]
              .groupby(['Year', 'WPP_RI']).size().reset_index(name='count'))

    ncols = 3
    nrows = int(np.ceil(len(wpp_in) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, code in enumerate(wpp_in):
        ax = axes[i]
        sub = annual[annual['WPP_RI'] == code].sort_values('Year')
        x, y = sub['Year'].values, sub['count'].values
        ax.plot(x, y, 'o--', color='#E03B3B', linewidth=1.5, markersize=6)

        if len(x) >= 2:
            slope, intercept, r, p, _ = stats.linregress(x, y)
            x_line = np.array([x.min(), x.max()])
            ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=1.2)
            ax.text(0.05, 0.92,
                    f'y = {slope:.1f}x + {intercept:.0f}\nR²={r**2:.3f}, p={p:.3f}',
                    transform=ax.transAxes, fontsize=8, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

        ax.set_title(f'WPP-RI {code}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Tahun')
        ax.set_ylabel('Jumlah Deteksi')
        ax.grid(True, linestyle='--', alpha=0.4)

    for j in range(len(wpp_in), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Tren Tahunan Deteksi Kapal per WPP-RI (QF_Detect=1)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig2_annual_trend.png')
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Tersimpan: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3: POLA BULANAN (SEASONALITY) PER TAHUN
# ══════════════════════════════════════════════════════════════════════════════
def fig3_monthly_seasonality(df):
    print("Membuat Fig 3: Pola Musiman Bulanan...")
    monthly = (df[df['WPP_RI'].isin(WPP_CODES)]
               .groupby(['Year', 'Month', 'WPP_RI']).size().reset_index(name='count'))
    years  = sorted(monthly['Year'].unique())
    wpp_in = [c for c in WPP_CODES if c in monthly['WPP_RI'].unique()]

    if not wpp_in or not years:
        print("  Data tidak cukup, skip.")
        return

    cmap_yr  = plt.get_cmap('plasma')
    yr_colors = {yr: cmap_yr(i / max(len(years) - 1, 1)) for i, yr in enumerate(years)}

    ncols = 3
    nrows = int(np.ceil(len(wpp_in) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, code in enumerate(wpp_in):
        ax = axes[i]
        sub_wpp = monthly[monthly['WPP_RI'] == code]
        for yr in years:
            sub_yr = sub_wpp[sub_wpp['Year'] == yr].sort_values('Month')
            if not sub_yr.empty:
                ax.plot(sub_yr['Month'], sub_yr['count'], 'o-',
                        color=yr_colors[yr], linewidth=1.4, markersize=5, label=str(yr))
        ax.set_title(f'WPP-RI {code}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Bulan')
        ax.set_ylabel('Jumlah Deteksi')
        ax.set_xticks(range(1, 13))
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize=7)

    for j in range(len(wpp_in), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Pola Musiman Bulanan Deteksi Kapal per WPP-RI',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig3_monthly_seasonality.png')
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Tersimpan: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4: HISTOGRAM DISTRIBUSI INTENSITAS (Rad_DNB)
# ══════════════════════════════════════════════════════════════════════════════
def fig4_radiance_histogram(df):
    print("Membuat Fig 4: Histogram Distribusi Rad_DNB...")
    wpp_in = [c for c in WPP_CODES if c in df['WPP_RI'].unique()]
    if not wpp_in:
        print("  Tidak ada data dalam WPP-RI, skip.")
        return

    ncols = 3
    nrows = int(np.ceil(len(wpp_in) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, code in enumerate(wpp_in):
        ax = axes[i]
        sub = df[df['WPP_RI'] == code]['Rad_DNB'].dropna()
        if sub.empty:
            continue
        log_vals = np.log1p(sub.clip(lower=0))
        ax.hist(log_vals, bins=30, color=WPP_COLOR[code],
                edgecolor='white', linewidth=0.5, alpha=0.85)
        median_val = np.exp(log_vals.median()) - 1
        ax.axvline(log_vals.median(), color='red', linestyle='--',
                   linewidth=1.2, label=f'Median: {median_val:.2f}')
        ax.set_title(f'WPP-RI {code} (n={len(sub)})', fontsize=10, fontweight='bold')
        ax.set_xlabel('log(1 + Rad_DNB) [nW/cm²/sr]')
        ax.set_ylabel('Frekuensi')
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.4)

    for j in range(len(wpp_in), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Distribusi Intensitas Cahaya Kapal (Rad_DNB) per WPP-RI',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig4_radiance_histogram.png')
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Tersimpan: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5: DENSITY HEATMAP SPASIAL
# ══════════════════════════════════════════════════════════════════════════════
def fig5_density_heatmap(df, wpp_geoms, indonesia_polys):
    print("Membuat Fig 5: Density Heatmap Spasial...")
    lons = df['Lon_DNB'].values
    lats = df['Lat_DNB'].values

    lon_bins = np.arange(90, 145, 0.5)
    lat_bins = np.arange(-12, 12, 0.5)
    H, xedges, yedges = np.histogram2d(lons, lats, bins=[lon_bins, lat_bins])
    H = H.T
    H_masked = np.ma.masked_where(H == 0, H)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_facecolor('#A8D5E8')

    draw_indonesia_land(ax, indonesia_polys)
    draw_wpp_boundaries(ax, wpp_geoms)

    vmax = max(H.max(), 2)
    im = ax.pcolormesh(xedges, yedges, H_masked,
                       cmap='YlOrRd', norm=mcolors.LogNorm(vmin=1, vmax=vmax),
                       alpha=0.85, zorder=6)
    plt.colorbar(im, ax=ax, label='Jumlah Deteksi per Grid 0.5°×0.5°', shrink=0.7)

    ax.set_xlim(90, 145)
    ax.set_ylim(-12, 12)
    ax.set_xlabel('Bujur (Longitude)', fontsize=11)
    ax.set_ylabel('Lintang (Latitude)', fontsize=11)
    ax.set_title('Density Heatmap Sebaran Kapal VIIRS (QF_Detect=1)\nGrid Spasial 0.5°×0.5°',
                 fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, zorder=1)

    out = os.path.join(FIG_DIR, 'fig5_density_heatmap.png')
    plt.tight_layout()
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Tersimpan: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6: LATITUDE-TIME DISTRIBUTION HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
def fig6_latitude_time(df):
    print("Membuat Fig 6: Latitude-Time Distribution...")
    df2 = df.copy()
    df2['YearMonth'] = pd.to_datetime(df2['Date']).dt.to_period('M').astype(str)
    df2['Lat_bin']   = pd.cut(df2['Lat_DNB'], bins=np.arange(-12, 13, 1), right=False)
    df2['Lat_mid']   = df2['Lat_bin'].apply(lambda x: x.mid if pd.notna(x) else np.nan)

    pivot = (df2.dropna(subset=['Lat_mid'])
             .groupby(['YearMonth', 'Lat_mid']).size().unstack(fill_value=0))

    if pivot.empty or pivot.shape[1] < 2:
        print("  Data tidak cukup untuk heatmap lat-time, skip.")
        return

    vmax = max(pivot.values.max(), 2)
    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(pivot.T.values, aspect='auto', cmap='plasma',
                   norm=mcolors.LogNorm(vmin=1, vmax=vmax), origin='lower')
    plt.colorbar(im, ax=ax, label='Jumlah Deteksi', shrink=0.8)

    ax.set_xticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.index, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(pivot.columns)))
    ax.set_yticklabels([f'{v:.0f}°' for v in pivot.columns], fontsize=8)
    ax.set_xlabel('Periode (Tahun-Bulan)', fontsize=11)
    ax.set_ylabel('Lintang (Latitude)', fontsize=11)
    ax.set_title('Distribusi Latitude-Waktu Deteksi Kapal VIIRS (QF_Detect=1)',
                 fontsize=13, fontweight='bold')

    out = os.path.join(FIG_DIR, 'fig6_latitude_time_heatmap.png')
    plt.tight_layout()
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Tersimpan: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    if not os.path.exists(FILTERED_CSV):
        print(f"ERROR: {FILTERED_CSV} tidak ditemukan.")
        print("Jalankan dulu: python process_viirs_wpp.py")
        return

    print(f"Memuat data: {FILTERED_CSV}")
    df = pd.read_csv(FILTERED_CSV, parse_dates=['Date'])
    print(f"Total baris    : {len(df)}")
    print(f"Rentang tanggal: {df['Date'].min()} s/d {df['Date'].max()}")
    print(f"Distribusi WPP :")
    print(df['WPP_RI'].value_counts().to_string())
    print()

    wpp_geoms       = load_wpp_geoms()
    indonesia_polys = get_indonesia_basemap()

    fig1_map(df, wpp_geoms, indonesia_polys)
    fig2_annual_trend(df)
    fig3_monthly_seasonality(df)
    fig4_radiance_histogram(df)
    fig5_density_heatmap(df, wpp_geoms, indonesia_polys)
    fig6_latitude_time(df)

    print("\n[SELESAI] Semua gambar tersimpan di output/figures/")
    print("Selanjutnya jalankan: python predict_xgboost.py")


if __name__ == '__main__':
    main()
