# Dokumentasi Visualisasi — VBD 2015

## Daftar Riset

| No | Tujuan Riset | Kode |
|----|-------------|------|
| 1 | Mencari pola penangkapan ikan di Core Fishing Area (CFA) tertentu | **R1** |
| 2 | Mencari spot lokasi penangkapan ikan (CFA) | **R2** |
| 3 | Clustering dari HDBSCAN | **R3** |
| 4 | Prediksi fishing ground | **R4** |
| 5 | Klasifikasi jenis boat berdasarkan brightness lampu | **R5** |
| 6 | Deteksi boat cahaya kecil | **R6** |
| 7 | Spesifikasi kapal | **R7** |
| 8 | Deteksi Illegal fishing | **R8** |

> **Catatan:** R3 (Clustering), R4 (Prediksi), dan R7 (Spesifikasi) belum termasuk di file ini.
> R3 & R4 akan dikerjakan di `visualisasi_ml.py` (terpisah).
> R7 tidak feasible dari data VIIRS (tidak ada dimensi/jenis kapal).

---

## Mapping File → Tujuan Riset

### 1. Pola Penangkapan Ikan (R1)

| File | Format | Keterangan | Interaktivitas |
|------|--------|------------|----------------|
| `pola_calendar_heatmap.html` | HTML | Heatmap jumlah kapal per hari (sumbu X = hari, Y = bulan). Melihat musim ramai vs sepi. | Hover lihat jumlah kapal per tanggal |
| `pola_monthly_trend.html` | HTML | Line chart trend bulanan per WPP. Bisa toggle WPP via legenda. | Klik legenda show/hide WPP, hover unified |
| `pola_jam_bulan.html` | HTML | Heatmap aktivitas jam vs bulan. Menjawab: jam berapa & bulan apa kapal paling aktif. | Hover lihat jumlah |
| `pola_qf_detect.html` | HTML | Stacked bar chart komposisi QF_Detect per WPP (Fishing vs Transit vs Lainnya). | Hover lihat detail |
| `pola_qf_detect.xlsx` | Excel | Tabel mentah QF_Detect per WPP untuk analisis lanjutan. | — |

**Interpretasi:**
- QF_Detect = 2, 4 → indikasi **fishing**
- QF_Detect = 1 → **transit**
- Perbandingan fishing vs transit per WPP menunjukkan WPP mana yang jadi prioritas penangkapan.
- Calendar heatmap menunjukkan pola musiman (bulan ramai vs sepi).

---

### 2. Spot Lokasi Penangkapan (CFA) (R2)

| File | Format | Keterangan | Interaktivitas |
|------|--------|------------|----------------|
| `spot_hotspot_top20.html` | HTML | Peta 20 grid cell (0.05°) dengan konsentrasi kapal tertinggi. Nomor = rank. | Hover lihat koordinat & jumlah |
| `spot_hotspot_top20.xlsx` | Excel | Tabel detail 20 hotspot (rank, lat, lon, jumlah kapal). | — |
| `spot_contour_density.html` | HTML | Contour density map — garis kontur konsentrasi kapal. | Hover lihat density |
| `spot_centroid_seasonal.html` | HTML | Peta pergeseran centroid (pusat aktivitas) per kuartal (Q1-Q4). | Hover lihat koordinat centroid |

**Interpretasi:**
- Top 20 hotspot bisa digunakan sebagai kandidat **Core Fishing Area**.
- Pergeseran centroid musiman menunjukkan pergerakan armada mengikuti musim ikan.
- Contour density melengkapi dengan visualisasi gradasi kepadatan.

---

### 3. Klasifikasi Boat Berdasarkan Brightness (R5)

| File | Format | Keterangan | Interaktivitas |
|------|--------|------------|----------------|
| `klas_histogram_radiance.html` | HTML | Histogram distribusi Rad_DNB (multimodal). Identifikasi kelas kapal berdasarkan tingkat kecerahan. | Hover lihat jumlah |
| `klas_box_radiance_qf.html` | HTML | Box plot radiance per QF_Detect. Sebaran brightness tiap kategori deteksi. | Hover lihat statistik |
| `klas_radiance_map.html` | HTML | Peta seluruh kapal dengan warna = brightness (Rad_DNB). | Hover lihat detail kapal |
| `klas_statistik_radiance.xlsx` | Excel | Statistik radiance per WPP (mean, median, std, min, max). | — |

**Interpretasi:**
- Histogram multimodal → ada beberapa kelas kapal berdasarkan brightness:
  - Mode rendah (~0-3): kapal kecil / cahaya redup
  - Mode menengah (~5-20): kapal sedang
  - Mode tinggi (~30+): kapal besar / terang
- Box plot menunjukkan distribusi brightness per QF_Detect.

---

### 4. Deteksi Boat Cahaya Kecil (R6)

| File | Format | Keterangan | Interaktivitas |
|------|--------|------------|----------------|
| `dim_boat_map.html` | HTML | Peta khusus kapal dengan radiance rendah (threshold ≤ 3.0). | Hover lihat detail |
| `dim_comparison_wpp.html` | HTML | Bar chart perbandingan jumlah kapal redup vs terang per WPP. | Hover lihat jumlah |
| `dim_perbandingan.xlsx` | Excel | Tabel perbandingan redup vs terang per WPP dengan persentase. | — |

**Interpretasi:**
- 191.707 kapal redup terdeteksi (~23.9% dari total).
- Threshold bisa diubah di script (`threshold = 3.0`).
- WPP dengan proporsi kapal redup tinggi → indikasi armada kecil tradisional.

---

### 5. Deteksi Illegal Fishing (R8)

| File | Format | Keterangan | Interaktivitas |
|------|--------|------------|----------------|
| `illegal_mpa_map.html` | HTML | Peta kapal yang terdeteksi di area MPA, dibedakan per MPA. Latar = semua kapal (abu-abu). | Klik legenda toggle MPA, hover lihat info |
| `illegal_flagging.html` | HTML | Tabel kapal mencurigakan (radiance tinggi di MPA + tanpa WPP di MPA). | — |
| `illegal_flagging.xlsx` | Excel | Data mentah flagging untuk analisis lanjutan. | — |
| `illegal_mpa_summary.xlsx` | Excel | Ringkasan tiap MPA (jumlah kapal, radiance, bounding box). | — |

**Interpretasi:**
- 23.726 kapal terdeteksi dengan data MPA, tersebar di **83 area MPA**.
- 56 kapal terflag mencurigakan (radiance tinggi di MPA).
- **Keterbatasan:** tanpa data VMS, deteksi illegal fishing hanya berdasarkan overlay spasial dengan MPA, bukan validasi izin kapal.

---

## Ringkasan File per Tujuan Riset (Statistik)

```
R1 — Pola Penangkapan      → pola_calendar_heatmap.html
                             pola_monthly_trend.html
                             pola_jam_bulan.html
                             pola_qf_detect.html / .xlsx

R2 — Spot CFA               → spot_hotspot_top20.html / .xlsx
                             spot_contour_density.html
                             spot_centroid_seasonal.html

R5 — Klasifikasi Boat       → klas_histogram_radiance.html
                             klas_box_radiance_qf.html
                             klas_radiance_map.html
                             klas_statistik_radiance.xlsx

R6 — Boat Cahaya Kecil      → dim_boat_map.html
                             dim_comparison_wpp.html / .xlsx

R8 — Illegal Fishing        → illegal_mpa_map.html
                             illegal_flagging.html / .xlsx
                             illegal_mpa_summary.xlsx
```

## Cara Menggunakan

1. Buka file `.html` di browser manapun (Chrome/Firefox/Edge).
2. Gunakan **scroll** untuk zoom, **drag** untuk pan pada peta.
3. **Hover** pada titik/bar/cell untuk melihat informasi detail.
4. **Klik legenda** untuk menampilkan/menyembunyikan layer tertentu.
5. File `.xlsx` bisa dibuka di Excel/LibreOffice untuk sorting & filter.

## Catatan

- Data: VIIRS DNB VBD (Vessel Boat Detection) — NOAA
- Tahun: 2015 (365 hari)
- Area: Perairan Indonesia
- Total kapal terdeteksi: ~1.3 juta, dengan WPP: ~803 ribu
- File output ada di: `/mnt/c/harr/research/vbd2015/output/`
- Script sumber: `visualisasi_statistik.py`
