"""
STEP 1: process_viirs_wpp.py (Refactored with GeoPandas)
----------------------------
Membaca semua file CSV VIIRS dari folder viirs/, filter QF_Detect==1,
lakukan spatial join (gpd.sjoin) ke WPP-RI shapefiles, simpan ke output/filtered_data.csv
Jalankan: python process_viirs_wpp.py
"""

import os
os.environ['SHAPE_RESTORE_SHX'] = 'YES'
import re
import pandas as pd
import geopandas as gpd
import time

# ─── Konfigurasi Path (Relative) ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
SHP_DIR   = os.path.join(BASE_DIR, 'shp')
VIIRS_DIR = os.path.join(BASE_DIR, 'viirs')
OUT_CSV   = os.path.join(BASE_DIR, 'output', 'filtered_data.csv')

def extract_date_from_filename(filename):
    """Ekstrak tanggal dari nama file, contoh: VBD_npp_d20240101_... -> 2024-01-01"""
    match = re.search(r'd(\d{4})(\d{2})(\d{2})', filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return None

def load_wpp_geometries(shp_dir):
    """Baca semua shapefile WPP-RI dan gabungkan menjadi satu GeoDataFrame."""
    print("Memuat shapefile WPP-RI...")
    gdfs = []
    
    if not os.path.exists(shp_dir):
        print(f"Directory tidak ditemukan: {shp_dir}")
        return None
        
    for fn in sorted(os.listdir(shp_dir)):
        if not fn.endswith('.shp'):
            continue
        m = re.search(r'WPP-RI\s+(\d+)', fn)
        wpp_code = m.group(1) if m else fn.replace('.shp', '')
        fp = os.path.join(shp_dir, fn)
        try:
            gdf = gpd.read_file(fp)
            # Karena beberapa SHP mungkin tidak punya kolom WPP_RI, kita tambahkan manual
            gdf['WPP_CODE'] = wpp_code
            gdfs.append(gdf[['WPP_CODE', 'geometry']])
            print(f"  [OK] WPP-RI {wpp_code} ({len(gdf)} poligon)")
        except Exception as e:
            print(f"  [ERR] {fn}: {e}")
            
    if not gdfs:
        return None
        
    combined_gdf = pd.concat(gdfs, ignore_index=True)
    
    # Set CRS ke WGS84 jika belum ada
    if combined_gdf.crs is None:
        combined_gdf = combined_gdf.set_crs("EPSG:4326")
    elif combined_gdf.crs.to_string() != "EPSG:4326":
        combined_gdf = combined_gdf.to_crs("EPSG:4326")
        
    return combined_gdf

def load_all_viirs(viirs_dir):
    """Baca semua CSV VIIRS di folder viirs/."""
    print("\nMemuat semua file VIIRS...")
    all_dfs = []
    
    if not os.path.exists(viirs_dir):
        print(f"Directory tidak ditemukan: {viirs_dir}")
        return None
        
    for fn in sorted(os.listdir(viirs_dir)):
        if not fn.endswith('.csv'):
            continue
        fp = os.path.join(viirs_dir, fn)
        date_str = extract_date_from_filename(fn)
        df = pd.read_csv(fp)

        # Pastikan kolom wajib ada
        required = {'Lat_DNB', 'Lon_DNB', 'Rad_DNB', 'QF_Detect'}
        if not required.issubset(df.columns):
            print(f"  [SKIP] {fn}: kolom tidak lengkap")
            continue

        # Tambahkan kolom tanggal jika belum ada
        if 'Date_Mscan' not in df.columns:
            df['Date_Mscan'] = date_str
        
        # Ekstrak tanggal
        df['Date_Mscan'] = pd.to_datetime(df['Date_Mscan'], errors='coerce')
        df['Year']  = df['Date_Mscan'].dt.year
        df['Month'] = df['Date_Mscan'].dt.month
        df['Day']   = df['Date_Mscan'].dt.day
        df['Date']  = df['Date_Mscan'].dt.date

        cols = ['Date', 'Year', 'Month', 'Day', 'Lat_DNB', 'Lon_DNB', 'Rad_DNB', 'QF_Detect']
        df = df[cols]
        all_dfs.append(df)
        print(f"  [OK] {fn}: {len(df)} baris")

    if not all_dfs:
        raise ValueError("Tidak ada file VIIRS yang berhasil dimuat.")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    return combined

def main():
    start_time = time.time()
    
    # 1. Muat WPP shapefile
    wpp_gdf = load_wpp_geometries(SHP_DIR)
    if wpp_gdf is None:
        print("Gagal memuat shapefile WPP-RI. Hentikan.")
        return

    # 2. Muat semua VIIRS
    df = load_all_viirs(VIIRS_DIR)
    if df is None:
        return
        
    print(f"\nTotal data sebelum filter: {len(df)} baris")

    # 3. Filter QF_Detect == 1
    df = df[df['QF_Detect'] == 1].reset_index(drop=True)
    print(f"Setelah filter QF_Detect==1: {len(df)} baris")

    # 4. Spatial join ke WPP-RI menggunakan GeoPandas (Sangat Cepat!)
    print("\nMelakukan spatial join (Point-in-Polygon)...")
    sjoin_start = time.time()
    
    # Konversi DataFrame Pandas ke GeoDataFrame
    viirs_gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df.Lon_DNB, df.Lat_DNB),
        crs="EPSG:4326"
    )
    
    # Lakukan spatial join (how='left' agar titik yang di luar tetap ada)
    joined_gdf = gpd.sjoin(viirs_gdf, wpp_gdf, how="left", predicate="within")
    
    # Mapping 'index_right' yang NaN menjadi 'Outside', yang terisi ambil 'WPP_CODE'
    joined_gdf['WPP_RI'] = joined_gdf['WPP_CODE'].fillna('Outside')
    
    # Hapus kolom geometri dan kolom sementara sjoin
    df_final = pd.DataFrame(joined_gdf.drop(columns=['geometry', 'index_right', 'WPP_CODE']))
    
    sjoin_time = time.time() - sjoin_start
    print(f"Spatial join selesai dalam {sjoin_time:.2f} detik.")

    # 5. Simpan
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df_final.to_csv(OUT_CSV, index=False)
    
    total_time = time.time() - start_time
    print(f"\nData tersimpan ke: {OUT_CSV}")
    print("\nRingkasan deteksi per WPP-RI:")
    print(df_final['WPP_RI'].value_counts().to_string())
    print(f"\n[SELESAI] Total waktu eksekusi: {total_time:.2f} detik.")
    print("Jalankan: python visualize_analysis.py")

if __name__ == '__main__':
    main()
