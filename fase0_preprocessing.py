"""
FASE    : PREPROCESSING
============================
Tujuan  : Menghasilkan file data bersih & teragregasi siap pakai untuk semua visualisasi
Input   : riset_viirs/viirs/[tahun]/*.csv  (17.8 juta baris VBD 2012-2026)
Output  : riset_viirs/output/
  1. pixel_grid_all.csv         -- semua piksel grid 0.01 deg, freq + radiance stats
  2. pixel_grid_L2.csv          -- subset piksel >= 22 hari (L2 threshold, default display)
  3. monthly_aggregated.csv     -- deteksi per tahun x bulan x WPP
  4. yearly_aggregated.csv      -- deteksi per tahun x WPP (2026 ditandai parsial)
  5. radiance_sample_log.csv    -- sample radiance dengan log-transform untuk distribusi

Filter yang diterapkan:
  - EEZ = "Indonesian Exclusive Economic Zone" (buang data Filipina/non-Indonesia)
  - Land_Mask != 1 (buang daratan)
  - 2026 ditandai is_partial=True (data hanya s/d Juli)
"""

import os
import sys
import glob
import csv
import math
from collections import defaultdict
import time

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

# ---- KONFIGURASI -----------------------------------------------------------
VIIRS_BASE  = r"e:\UNIVERSITAS PENDIDIKAN INDONESIA\PENELITIAN\FISHING GROUND PATTERN\riset_viirs\viirs"
OUTPUT_DIR  = r"e:\UNIVERSITAS PENDIDIKAN INDONESIA\PENELITIAN\FISHING GROUND PATTERN\riset_viirs\output"
GRID_DEG    = 0.01   # resolusi grid ~1.1 km

# Threshold level (berbasis data empiris - ThresholdEmpiris.md)
THRESHOLD_L1 = 10    # P80  -- Eksplorasi
THRESHOLD_L2 = 22    # P90  -- DEFAULT display
THRESHOLD_L3 = 37    # P95  -- Significant / paper
THRESHOLD_L4 = 68    # P98  -- Hotspot terkonfirmasi
THRESHOLD_L5 = 101   # P99  -- Core / inti permanen
THRESHOLD_L6 = 146   # P99.5 -- Apex / main figure

EEZ_VALID   = "Indonesian Exclusive Economic Zone"
YEARS       = list(range(2012, 2027))
YEARS_FULL  = list(range(2012, 2026))   # tahun penuh
YEAR_PARTIAL = 2026                     # parsial s/d Juli

MAX_RAD_SAMPLE = 1_000_000  # max sampel radiance untuk file distribusi

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- FUNGSI BANTU ----------------------------------------------------------

def snap_grid(val, deg=GRID_DEG):
    return round(math.floor(val / deg) * deg + deg / 2, 6)

def safe_float(v, default=0.0):
    try:
        return float(v) if v else default
    except (ValueError, TypeError):
        return default

def safe_int(v, default=0):
    try:
        return int(v) if v else default
    except (ValueError, TypeError):
        return default

def log1p_safe(v):
    """log(1 + v) -- stabil saat v=0, tidak undefined."""
    if v < 0:
        return 0.0
    return math.log1p(v)

SEP = "=" * 72
print(SEP)
print("FASE 0: PREPROCESSING WAJIB -- VBD INDONESIA 2012-2026")
print(SEP)
print(f"\nGrid    : {GRID_DEG} deg (~{GRID_DEG*111:.1f} km)")
print(f"Filter  : EEZ = '{EEZ_VALID}'")
print(f"L2 Thr  : >= {THRESHOLD_L2} hari/piksel (default display)")
print(f"Output  : {OUTPUT_DIR}")
print()

t0 = time.time()

# ============================================================================
# LANGKAH 1: BACA SEMUA DATA + AGREGASI
# ============================================================================
print("[1/6] Membaca dan mengagregasi data CSV...")

# Struktur data yang dikumpulkan
pixel_freq   = defaultdict(int)              # {(lat,lon): total_hari}
pixel_radsum = defaultdict(float)            # {(lat,lon): sum radiance}
pixel_radmax = defaultdict(float)            # {(lat,lon): max radiance}
pixel_radcnt = defaultdict(int)              # {(lat,lon): n deteksi}
pixel_logsum = defaultdict(float)            # {(lat,lon): sum log1p(rad)}
pixel_wpp    = defaultdict(lambda: defaultdict(int))  # {(lat,lon): {wpp: count}}

# monthly: {(year, month, wpp): count}
monthly      = defaultdict(int)
monthly_rad  = defaultdict(float)
monthly_cnt  = defaultdict(int)

# yearly: {(year, wpp): count}
yearly       = defaultdict(int)
yearly_rad   = defaultdict(float)
yearly_cnt   = defaultdict(int)

# Sample radiance untuk distribusi
rad_sample   = []

total_rows   = 0
skip_eez     = 0
skip_land    = 0
total_files  = 0
error_files  = 0

for year in YEARS:
    year_dir = os.path.join(VIIRS_BASE, str(year))
    if not os.path.isdir(year_dir):
        print(f"  [SKIP] {year}: direktori tidak ditemukan")
        continue

    csv_files = sorted(glob.glob(os.path.join(year_dir, "*.csv")))
    year_rows = year_skip_eez = year_skip_land = 0

    for fpath in csv_files:
        if os.path.getsize(fpath) < 500:
            continue
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        # --- FILTER 1: EEZ Indonesia ---
                        eez = (row.get("EEZ") or "").strip()
                        if eez != EEZ_VALID:
                            skip_eez += 1
                            year_skip_eez += 1
                            continue

                        # --- FILTER 2: Daratan ---
                        land = safe_int(row.get("Land_Mask"), 3)
                        if land == 1:
                            skip_land += 1
                            year_skip_land += 1
                            continue

                        lat = safe_float(row.get("Lat_DNB"))
                        lon = safe_float(row.get("Lon_DNB"))
                        rad = safe_float(row.get("Rad_DNB"))
                        wpp = (row.get("FMZ") or "Unknown").strip()

                        # Ekstrak bulan dari Date_Mscan (format: "YYYY-MM-DD HH:MM:SS")
                        date_str = row.get("Date_Mscan") or row.get("Date_Proc") or ""
                        try:
                            month = int(date_str[5:7]) if len(date_str) >= 7 else 0
                        except (ValueError, IndexError):
                            month = 0

                        key = (snap_grid(lat), snap_grid(lon))

                        # Pixel aggregation
                        pixel_freq[key]   += 1
                        pixel_radsum[key] += rad
                        pixel_radcnt[key] += 1
                        pixel_logsum[key] += log1p_safe(rad)
                        if rad > pixel_radmax[key]:
                            pixel_radmax[key] = rad
                        pixel_wpp[key][wpp] += 1

                        # Monthly aggregation
                        mkey = (year, month, wpp)
                        monthly[mkey]     += 1
                        monthly_rad[mkey] += rad
                        monthly_cnt[mkey] += 1

                        # Yearly aggregation
                        ykey = (year, wpp)
                        yearly[ykey]     += 1
                        yearly_rad[ykey] += rad
                        yearly_cnt[ykey] += 1

                        # Sample radiance
                        if len(rad_sample) < MAX_RAD_SAMPLE:
                            rad_sample.append((year, month, wpp, rad, log1p_safe(rad)))

                        total_rows  += 1
                        year_rows   += 1

                    except (ValueError, KeyError):
                        continue

            total_files += 1
        except Exception:
            error_files += 1

    elapsed = time.time() - t0
    partial_flag = " [PARSIAL]" if year == YEAR_PARTIAL else ""
    print(f"  {year}{partial_flag}: {len(csv_files):4d} files | "
          f"{year_rows:>9,} valid | skip_eez={year_skip_eez:,} "
          f"skip_land={year_skip_land:,} | {elapsed:.0f}s")

print(f"\n  Total baris valid (Indonesia) : {total_rows:,}")
print(f"  Dibuang (non-Indonesia/EEZ)   : {skip_eez:,}")
print(f"  Dibuang (daratan)             : {skip_land:,}")
print(f"  Piksel unik (grid)            : {len(pixel_freq):,}")
print(f"  File proses                   : {total_files:,} | error: {error_files}")

# ============================================================================
# LANGKAH 2: OUTPUT pixel_grid_all.csv
# ============================================================================
print("\n[2/6] Menulis pixel_grid_all.csv ...")

pga_path = os.path.join(OUTPUT_DIR, "pixel_grid_all.csv")
written_pga = 0

with open(pga_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "lat_grid", "lon_grid",
        "freq_days",        # jumlah hari terdeteksi (frekuensi kunjungan)
        "rad_mean",         # rata-rata radiance
        "rad_max",          # radiance maksimum
        "log_rad_mean",     # mean dari log1p(radiance)
        "wpp_dominant",     # WPP paling sering (dominant)
        # Flag threshold membership
        "is_L1",            # >= 10 hari
        "is_L2",            # >= 22 hari (DEFAULT)
        "is_L3",            # >= 37 hari
        "is_L4",            # >= 68 hari
        "is_L5",            # >= 101 hari
        "is_L6",            # >= 146 hari
    ])

    for key, freq in pixel_freq.items():
        lat_g, lon_g = key
        cnt  = pixel_radcnt[key] or 1
        rmean = pixel_radsum[key] / cnt
        rmax  = pixel_radmax[key]
        lmean = pixel_logsum[key] / cnt

        # Dominant WPP
        wpp_dict = pixel_wpp[key]
        dom_wpp  = max(wpp_dict, key=wpp_dict.get) if wpp_dict else "Unknown"

        w.writerow([
            lat_g, lon_g,
            freq,
            round(rmean, 4),
            round(rmax, 4),
            round(lmean, 6),
            dom_wpp,
            1 if freq >= THRESHOLD_L1 else 0,
            1 if freq >= THRESHOLD_L2 else 0,
            1 if freq >= THRESHOLD_L3 else 0,
            1 if freq >= THRESHOLD_L4 else 0,
            1 if freq >= THRESHOLD_L5 else 0,
            1 if freq >= THRESHOLD_L6 else 0,
        ])
        written_pga += 1

print(f"  Ditulis: {written_pga:,} baris -> {pga_path}")

# ============================================================================
# LANGKAH 3: OUTPUT pixel_grid_L2.csv (threshold display default)
# ============================================================================
print("\n[3/6] Menulis pixel_grid_L2.csv (>= 22 hari / P90) ...")

pgl2_path = os.path.join(OUTPUT_DIR, "pixel_grid_L2.csv")
written_l2 = 0

with open(pgl2_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "lat_grid", "lon_grid",
        "freq_days", "rad_mean", "rad_max", "log_rad_mean",
        "wpp_dominant",
        "is_L3", "is_L4", "is_L5", "is_L6",
    ])

    for key, freq in pixel_freq.items():
        if freq < THRESHOLD_L2:
            continue
        lat_g, lon_g = key
        cnt   = pixel_radcnt[key] or 1
        rmean = pixel_radsum[key] / cnt
        rmax  = pixel_radmax[key]
        lmean = pixel_logsum[key] / cnt
        wpp_dict = pixel_wpp[key]
        dom_wpp  = max(wpp_dict, key=wpp_dict.get) if wpp_dict else "Unknown"

        w.writerow([
            lat_g, lon_g,
            freq,
            round(rmean, 4),
            round(rmax, 4),
            round(lmean, 6),
            dom_wpp,
            1 if freq >= THRESHOLD_L3 else 0,
            1 if freq >= THRESHOLD_L4 else 0,
            1 if freq >= THRESHOLD_L5 else 0,
            1 if freq >= THRESHOLD_L6 else 0,
        ])
        written_l2 += 1

print(f"  Ditulis: {written_l2:,} baris -> {pgl2_path}")
print(f"  (dari {written_pga:,} piksel total -> {100*written_l2/written_pga:.1f}% lolos L2)")

# ============================================================================
# LANGKAH 4: OUTPUT monthly_aggregated.csv
# ============================================================================
print("\n[4/6] Menulis monthly_aggregated.csv ...")

mon_path = os.path.join(OUTPUT_DIR, "monthly_aggregated.csv")
written_mon = 0

with open(mon_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "year", "month", "wpp",
        "detection_count",   # total baris VBD (kapal-hari)
        "rad_mean",          # rata-rata radiance bulan ini
        "log_rad_mean",      # rata-rata log1p(radiance)
        "is_partial_year",   # 1 jika tahun 2026
    ])

    for (year, month, wpp), count in sorted(monthly.items()):
        cnt  = monthly_cnt[(year, month, wpp)] or 1
        rmean = monthly_rad[(year, month, wpp)] / cnt
        lmean = math.log1p(rmean) if rmean >= 0 else 0
        w.writerow([
            year, month, wpp,
            count,
            round(rmean, 4),
            round(lmean, 6),
            1 if year == YEAR_PARTIAL else 0,
        ])
        written_mon += 1

print(f"  Ditulis: {written_mon:,} baris -> {mon_path}")

# ============================================================================
# LANGKAH 5: OUTPUT yearly_aggregated.csv
# ============================================================================
print("\n[5/6] Menulis yearly_aggregated.csv ...")

yr_path = os.path.join(OUTPUT_DIR, "yearly_aggregated.csv")
written_yr = 0

with open(yr_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "year", "wpp",
        "detection_count",
        "rad_mean",
        "log_rad_mean",
        "is_partial_year",   # 1 jika 2026 (data parsial s/d Juli)
        "coverage_note",
    ])

    for (year, wpp), count in sorted(yearly.items()):
        cnt   = yearly_cnt[(year, wpp)] or 1
        rmean = yearly_rad[(year, wpp)] / cnt
        lmean = math.log1p(rmean) if rmean >= 0 else 0
        note  = "Partial (Jan-Jul only)" if year == YEAR_PARTIAL else "Full year"
        w.writerow([
            year, wpp,
            count,
            round(rmean, 4),
            round(lmean, 6),
            1 if year == YEAR_PARTIAL else 0,
            note,
        ])
        written_yr += 1

print(f"  Ditulis: {written_yr:,} baris -> {yr_path}")

# ============================================================================
# LANGKAH 6: OUTPUT radiance_sample_log.csv
# ============================================================================
print("\n[6/6] Menulis radiance_sample_log.csv ...")

rad_path = os.path.join(OUTPUT_DIR, "radiance_sample_log.csv")

with open(rad_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "year", "month", "wpp",
        "radiance_raw",     # nilai asli (nW/cm2/sr)
        "radiance_log1p",   # log(1 + radiance) -- untuk visualisasi distribusi
        # Kelas radiance empiris (dari ThresholdEmpiris.md)
        "rad_class",        # "tradisional" / "menengah" / "industri" / "anomali"
    ])

    def rad_class(r):
        if r < 10:
            return "tradisional"
        elif r < 100:
            return "menengah"
        elif r < 1000:
            return "industri"
        else:
            return "anomali"

    for year, month, wpp, rad, log_rad in rad_sample:
        w.writerow([
            year, month, wpp,
            round(rad, 4),
            round(log_rad, 6),
            rad_class(rad),
        ])

print(f"  Ditulis: {len(rad_sample):,} baris -> {rad_path}")

# ============================================================================
# LAPORAN AKHIR
# ============================================================================
elapsed = time.time() - t0

print()
print(SEP)
print("FASE 0 SELESAI -- RINGKASAN OUTPUT")
print(SEP)
print(f"\nDurasi total : {elapsed:.0f} detik")
print(f"\nFile output di: {OUTPUT_DIR}")
print()
print(f"  1. pixel_grid_all.csv   -- {written_pga:>9,} piksel (semua, tanpa threshold)")
print(f"  2. pixel_grid_L2.csv    -- {written_l2:>9,} piksel (>= 22 hari / P90)")
print(f"  3. monthly_aggregated   -- {written_mon:>9,} baris (tahun x bulan x WPP)")
print(f"  4. yearly_aggregated    -- {written_yr:>9,} baris (tahun x WPP, 2026=parsial)")
print(f"  5. radiance_sample_log  -- {len(rad_sample):>9,} sampel (dengan log1p + kelas)")
print()
print("CATATAN PENTING:")
print(f"  - Data non-Indonesia dibuang : {skip_eez:,} baris")
print(f"  - Data daratan dibuang       : {skip_land:,} baris")
print(f"  - Baris Indonesia valid      : {total_rows:,}")
print(f"  - Tahun 2026 ditandai parsial (s/d Juli) di semua file")
print(f"  - radiance_log1p = log(1 + radiance) -- gunakan kolom ini untuk semua plot distribusi")
print()
print("SIAP untuk Gelombang 1 Visualisasi:")
print("  -> D2: ECDF     (gunakan pixel_grid_all.csv, kolom freq_days)")
print("  -> A1: Calendar Heatmap (gunakan monthly_aggregated.csv)")
print("  -> V2: Pareto WPP       (gunakan yearly_aggregated.csv)")
print("  -> V3: Trend Line       (gunakan yearly_aggregated.csv)")
print("  -> D1: Radiance Hist    (gunakan radiance_sample_log.csv, kolom radiance_log1p)")
print(SEP)
