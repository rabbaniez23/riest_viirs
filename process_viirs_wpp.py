"""
STEP 1: process_viirs_wpp.py
----------------------------
Membaca semua file CSV VIIRS dari folder viirs/, filter QF_Detect==1,
lakukan spatial join ke WPP-RI shapefiles, simpan ke output/filtered_data.csv
Jalankan: python process_viirs_wpp.py
"""

import os
import re
import pandas as pd
import shapefile
from shapely.geometry import Point, shape
from shapely.prepared import prep
from shapely.ops import unary_union

# ─── Konfigurasi Path ─────────────────────────────────────────────────────────
SHP_DIR   = 'd:/Pekerjaan/riset/pertemuan4/shp'
VIIRS_DIR = 'd:/Pekerjaan/riset/pertemuan4/viirs'
OUT_CSV   = 'd:/Pekerjaan/riset/pertemuan4/output/filtered_data.csv'

def extract_date_from_filename(filename):
    """Ekstrak tanggal dari nama file, contoh: VBD_npp_d20240101_... -> 2024-01-01"""
    match = re.search(r'd(\d{4})(\d{2})(\d{2})', filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return None

def load_wpp_geometries(shp_dir):
    """Baca semua shapefile WPP-RI, kembalikan dict {kode_wpp: geometri}."""
    print("Memuat shapefile WPP-RI...")
    wpp_geoms = {}
    for fn in sorted(os.listdir(shp_dir)):
        if not fn.endswith('.shp'):
            continue
        m = re.search(r'WPP-RI\s+(\d+)', fn)
        wpp_code = m.group(1) if m else fn.replace('.shp', '')
        fp = os.path.join(shp_dir, fn)
        try:
            sf = shapefile.Reader(shp=fp)
            geom_list = [shape(s) for s in sf.shapes() if s.shapeType != 0]
            if not geom_list:
                continue
            wpp_geoms[wpp_code] = geom_list[0] if len(geom_list) == 1 else unary_union(geom_list)
            print(f"  [OK] WPP-RI {wpp_code} ({wpp_geoms[wpp_code].geom_type})")
        except Exception as e:
            print(f"  [ERR] {fn}: {e}")
    return wpp_geoms

def load_all_viirs(viirs_dir):
    """
    Baca semua CSV VIIRS di folder viirs/.
    Tangani dua format berbeda:
      - Format lama (2024): kolom minimal [Lat_DNB, Lon_DNB, Rad_DNB, QF_Detect]
      - Format lengkap (2025): banyak kolom, termasuk Date_Mscan
    """
    print("\nMemuat semua file VIIRS...")
    all_dfs = []
    for fn in sorted(os.listdir(viirs_dir)):
        if not fn.endswith('.csv'):
            continue
        fp = os.path.join(viirs_dir, fn)
        date_str = extract_date_from_filename(fn)
        df = pd.read_csv(fp)

        # Pastikan kolom wajib ada
        required = {'Lat_DNB', 'Lon_DNB', 'Rad_DNB', 'QF_Detect'}
        if not required.issubset(df.columns):
            print(f"  [SKIP] {fn}: kolom tidak lengkap {df.columns.tolist()}")
            continue

        # Tambahkan kolom tanggal jika belum ada
        if 'Date_Mscan' not in df.columns:
            df['Date_Mscan'] = date_str
        
        # Ekstrak tanggal ke kolom Year, Month, Day
        df['Date_Mscan'] = pd.to_datetime(df['Date_Mscan'], errors='coerce')
        df['Year']  = df['Date_Mscan'].dt.year
        df['Month'] = df['Date_Mscan'].dt.month
        df['Day']   = df['Date_Mscan'].dt.day
        df['Date']  = df['Date_Mscan'].dt.date

        # Simpan hanya kolom yang dibutuhkan
        cols = ['Date', 'Year', 'Month', 'Day', 'Lat_DNB', 'Lon_DNB', 'Rad_DNB', 'QF_Detect']
        df = df[cols]
        all_dfs.append(df)
        print(f"  [OK] {fn}: {len(df)} baris")

    if not all_dfs:
        raise ValueError("Tidak ada file VIIRS yang berhasil dimuat.")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    return combined

def assign_wpp(df, wpp_geoms):
    """Tentukan WPP-RI untuk setiap titik deteksi (Point-in-Polygon)."""
    print("\nMenyiapkan prepared geometries...")
    prepared = {code: prep(geom) for code, geom in wpp_geoms.items()}

    print(f"Mencocokkan {len(df)} titik ke WPP-RI...")
    results = []
    for i, row in df.iterrows():
        if i % 500 == 0:
            print(f"  ... {i}/{len(df)} titik diproses")
        p = Point(row['Lon_DNB'], row['Lat_DNB'])
        wpp = 'Outside'
        for code, prep_geom in prepared.items():
            if prep_geom.contains(p):
                wpp = code
                break
        results.append(wpp)
    
    df['WPP_RI'] = results
    return df

def main():
    # 1. Muat WPP shapefile
    wpp_geoms = load_wpp_geometries(SHP_DIR)
    if not wpp_geoms:
        print("Gagal memuat shapefile WPP-RI. Hentikan.")
        return

    # 2. Muat semua VIIRS
    df = load_all_viirs(VIIRS_DIR)
    print(f"\nTotal data sebelum filter: {len(df)} baris")

    # 3. Filter QF_Detect == 1
    df = df[df['QF_Detect'] == 1].reset_index(drop=True)
    print(f"Setelah filter QF_Detect==1: {len(df)} baris")

    # 4. Spatial join ke WPP-RI
    df = assign_wpp(df, wpp_geoms)

    # 5. Simpan
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nData tersimpan ke: {OUT_CSV}")
    print("\nRingkasan deteksi per WPP-RI:")
    print(df['WPP_RI'].value_counts().to_string())
    print("\n[SELESAI] Jalankan: python visualize_analysis.py")

if __name__ == '__main__':
    main()
