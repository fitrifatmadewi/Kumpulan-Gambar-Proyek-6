# Laporan Proyek Pertama Machine Learning Terapan
## Analisis Data Loan Approval Classification
### Oleh Fitri Fatma Dewi (MC004D5X1425)
---

## **I. Domain Proyek: Finansial – Risiko Kredit dan Inklusi Keuangan**

Risiko kredit merupakan aspek krusial dalam sistem keuangan karena berhubungan langsung dengan kelangsungan bisnis lembaga keuangan dan stabilitas ekonomi. Ketidakmampuan dalam mengelola risiko ini dapat menyebabkan tingginya *non-performing loans (NPL)* yang berdampak pada kerugian finansial institusi dan berkurangnya akses masyarakat terhadap layanan keuangan.

Menurut World Bank (2021), inklusi keuangan tetap menjadi tantangan di berbagai negara berkembang, di mana individu dengan riwayat kredit buruk atau tanpa riwayat sama sekali sulit mengakses pinjaman \[1]. Oleh karena itu, diperlukan sistem yang mampu memprediksi kelayakan kredit secara objektif dan efisien.

Penelitian terkini menegaskan bahwa pendekatan machine learning mampu meningkatkan akurasi dalam memprediksi risiko kredit dibandingkan metode konvensional \[2]. Dengan menggunakan data demografis, ekonomi, dan histori kredit, model machine learning dapat mengidentifikasi pola risiko secara lebih mendalam dan real-time.

Proyek ini berfokus pada pengembangan model klasifikasi untuk memprediksi status persetujuan pinjaman berdasarkan berbagai karakteristik peminjam. Model ini diharapkan dapat menjadi alat bantu pengambilan keputusan bagi lembaga keuangan untuk menyeimbangkan antara manajemen risiko dan perluasan akses kredit secara inklusif.

---

### Referensi:

\[1] World Bank. (2021). *Financial Inclusion*. Retrieved May 20, 2025, from [https://www.worldbank.org/en/topic/financialinclusion](https://www.worldbank.org/en/topic/financialinclusion)

\[2] Lessmann, S., Baesens, B., Seow, H. V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. *European Journal of Operational Research*, 247(1), 124–136. [https://doi.org/10.1016/j.ejor.2015.05.030](https://doi.org/10.1016/j.ejor.2015.05.030)

---

## **II. Business Understanding**

### II.a. Problem Statements

Lembaga keuangan menghadapi tantangan dalam menyaring pemohon pinjaman secara cepat dan akurat, khususnya dalam konteks risiko gagal bayar. Untuk mengatasi permasalahan tersebut, proyek ini mengajukan beberapa pertanyaan utama:

1. **Pernyataan Masalah 1:**
   Fitur-fitur apa yang paling berpengaruh dalam menentukan kelayakan persetujuan pinjaman bagi seorang pemohon?

2. **Pernyataan Masalah 2:**
   Bagaimana membangun model prediksi klasifikasi biner (approve/tolak) yang akurat berdasarkan fitur terpilih?

3. **Pernyataan Masalah 3:**
   Bagaimana memastikan model dapat melakukan generalisasi yang baik terhadap data baru yang belum pernah dilihat sebelumnya?

---

### II.b. Goals

Untuk menjawab permasalahan di atas, proyek ini memiliki tujuan sebagai berikut:

1. **Jawaban Pernyataan Masalah 1:**
   Melakukan seleksi fitur untuk mengidentifikasi fitur-fitur yang paling berpengaruh dalam menentukan kelayakan persetujuan pinjaman.

2. **Jawaban Pernyataan Masalah 2:**
   Mengembangkan model klasifikasi yang mampu memprediksi status persetujuan pinjaman berdasarkan fitur terpilih.

3. **Jawaban Pernyataan Masalah 3:**
   Melakukan evaluasi dan validasi model untuk memastikan performa optimal dalam hal akurasi, presisi, dan kemampuan generalisasi.

---

### II.c. Solution Statements

Untuk mencapai tujuan proyek, solusi yang akan diimplementasikan meliputi:

1. **Analisis Feature Importance:**
   Melakukan analisis fitur yang paling mempengaruhi berdasarkan nilai importance fitur dari model Random Forest.

2. **Eksperimen Beberapa Algoritma Klasifikasi:**
   Membangun dan membandingkan performa dari beberapa algoritma klasifikasi seperti:
   * Logistic Regression
   * XGBoost

3. **Evaluasi Berdasarkan Metrik yang Terukur:**
   Menggunakan metrik evaluasi seperti:
   * **Accuracy** untuk mengukur tingkat kebenaran prediksi secara keseluruhan.
   * **Precision, Recall, dan F1-Score** untuk mengevaluasi ketepatan dan kelengkapan model dalam mendeteksi pinjaman yang layak.
   * **ROC-AUC Curve** untuk menilai kemampuan model dalam membedakan kelas.
   * **Confussion Matrix** untuk memberikan gambaran detail tentang performa klasifikasi model, dengan menunjukkan jumlah prediksi benar dan salah pada setiap kelas.
---


## **III. Data Understanding**

### III. a. Tentang Data
Proyek ini menggunakan dataset **Loan Approval Classification Data** yang tersedia secara publik di [Kaggle](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data). Dataset ini dikembangkan untuk keperluan klasifikasi status persetujuan pinjaman berdasarkan informasi demografis dan karakteristik keuangan peminjam. Dataset ini merupakan hasil augmentasi dari data asli dengan teknik SMOTENC agar seimbang antar kelas target.

Dataset ini terdiri dari **45.000 observasi dan 14 fitur**, termasuk fitur numerik, kategorikal, dan Fitur target.

🔗 **Link Dataset:** [https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)

---

#### **Fitur pada Dataset**

| Fitur                         | Tipe                 | Deskripsi                                              |
| -------------------------------- | -------------------- | ------------------------------------------------------ |
| `person_age`                     | Numerik              | Usia peminjam                                          |
| `person_gender`                  | Kategorikal          | Jenis kelamin peminjam                                 |
| `person_education`               | Kategorikal          | Tingkat pendidikan terakhir                            |
| `person_income`                  | Numerik              | Pendapatan tahunan (USD)                               |
| `person_emp_exp`                 | Numerik              | Lama pengalaman kerja (tahun)                          |
| `person_home_ownership`          | Kategorikal          | Status kepemilikan tempat tinggal                      |
| `loan_intent`                    | Kategorikal          | Tujuan pinjaman (misal: pendidikan, pernikahan, rumah) |
| `loan_amnt`                      | Numerik              | Jumlah pinjaman yang diajukan                          |
| `loan_int_rate`                  | Numerik              | Suku bunga pinjaman (%)                                |
| `loan_percent_income`            | Numerik              | Persentase pinjaman terhadap pendapatan                |
| `cb_person_cred_hist_length`     | Numerik              | Panjang histori kredit (tahun)                         |
| `credit_score`                   | Numerik              | Skor kredit dari biro kredit                           |
| `previous_loan_defaults_on_file` | Kategorikal          | Riwayat gagal bayar pinjaman sebelumnya (Yes/No)       |
| `loan_status`                    | Kategorikal (Target) | Status pinjaman: 1 = Disetujui, 0 = Ditolak            |

---

### III.b. Exploratory Data Analysis (EDA)

#### **Statistika Deskriptif untuk Fitur Numerik**
Pada tahap awal dilakukan analisis statistika deskriptif untuk menggambarkan karakteristik data pinjaman. Analisis ini mencakup nilai-nilai statistik seperti rata-rata (mean), standar deviasi, serta nilai minimum dan maksimum dari setiap variabel numerik yang dianalisis. Analisis ini bertujuan memberikan gambaran awal terhadap pola dan variasi data sebelum dilakukan analisis lebih lanjut. 

| Statistik | person_age | person_income | person_emp_exp | loan_amnt  | loan_int_rate | loan_percent_income | cb_person_cred_hist_length | credit_score | loan_status |
|-----------|------------|---------------|---------------|------------|---------------|---------------------|----------------------------|--------------|-------------|
| count     | 45000      | 45000         | 45000         | 45000      | 45000         | 45000               | 45000                      | 45000        | 45000       |
| mean      | 27.76      | 80319.05      | 5.41          | 9583.16    | 11.01         | 0.14                | 5.87                       | 632.61       | 0.22        |
| std       | 6.05       | 80422.50      | 6.06          | 6314.89    | 2.98          | 0.09                | 3.88                       | 50.44        | 0.42        |
| min       | 20.00      | 8000.00       | 0.00          | 500.00     | 5.42          | 0.00                | 2.00                       | 390.00       | 0.00        |
| 25%       | 24.00      | 47204.00      | 1.00          | 5000.00    | 8.59          | 0.07                | 3.00                       | 601.00       | 0.00        |
| 50%       | 26.00      | 67048.00      | 4.00          | 8000.00    | 11.01         | 0.12                | 4.00                       | 640.00       | 0.00        |
| 75%       | 30.00      | 95789.25      | 8.00          | 12237.25   | 12.99         | 0.19                | 8.00                       | 670.00       | 0.00        |
| max       | 144.00     | 7200766.00    | 125.00        | 35000.00   | 20.00         | 0.66                | 30.00                      | 850.00       | 1.00        |

Berdasarkan statistika deskriptif di atas didapatkan beberapa informasi penting, antara lain:

1. **Usia Pemohon (person\_age)**
   Rata-rata usia pemohon adalah sekitar 27,76 tahun dengan standar deviasi 6,05 tahun, menunjukkan mayoritas pemohon berasal dari kalangan usia muda. Namun, terdapat nilai maksimum yang tidak wajar yaitu 144 tahun, yang kemungkinan merupakan kesalahan input atau outlier.

2. **Pendapatan Pemohon (person\_income)**
   Rata-rata pendapatan tahunan pemohon sebesar 80.319,05, namun dengan standar deviasi yang tinggi (sekitar 80.422,50), mengindikasikan adanya variasi pendapatan yang sangat besar. Nilai maksimum sebesar 7,2 juta sangat ekstrem dibandingkan dengan kuartil atas (\~95 ribu), yang juga menunjukkan kemungkinan outlier.

3. **Pengalaman Kerja (person\_emp\_exp)**
   Pengalaman kerja rata-rata adalah 5,41 tahun dengan distribusi yang sangat lebar (maksimum 125 tahun), menunjukkan kemungkinan data ekstrem atau tidak valid pada variabel ini.

4. **Jumlah Pinjaman (loan\_amnt)**
   Rata-rata pinjaman yang diajukan adalah sekitar 9.583,16 dengan standar deviasi sekitar 6.314,89. Nilai maksimum mencapai 35.000, menunjukkan variasi kebutuhan pinjaman yang cukup besar.

5. **Suku Bunga Pinjaman (loan\_int\_rate)**
   Suku bunga pinjaman bervariasi antara 5,42% hingga 20%, dengan rata-rata 11%, yang mencerminkan rentang suku bunga yang cukup luas dalam portofolio pinjaman ini.

6. **Persentase Pendapatan untuk Cicilan (loan\_percent\_income)**
   Proporsi pendapatan yang digunakan untuk membayar pinjaman berkisar dari 0% hingga 66%, dengan median sebesar 12%. Ini menunjukkan bahwa sebagian besar pemohon masih dalam batas rasio pembayaran pinjaman yang sehat, namun terdapat individu yang berisiko secara finansial.

7. **Lama Riwayat Kredit (cb\_person\_cred\_hist\_length)**
   Rata-rata panjang riwayat kredit adalah sekitar 5,87 tahun, dengan variasi hingga 30 tahun. Median berada di angka 4 tahun, menunjukkan banyak pemohon masih dalam tahap awal riwayat kredit.

8. **Skor Kredit (credit\_score)**
   Rata-rata skor kredit berada di angka 632,6 dengan rentang antara 390 hingga 850. Sebagian besar pemohon memiliki skor yang mendekati standar minimum kelayakan kredit (sekitar 640), yang penting untuk evaluasi risiko.

9. **Status Pinjaman (loan\_status)**
   Variabel ini bersifat biner, dengan nilai rata-rata 0,222, menunjukkan bahwa sekitar 22,2% pemohon gagal membayar pinjaman (loan\_status = 1), sementara sisanya sebesar 77,8% melunasi pinjaman dengan lancar.

Temuan seperti nilai ekstrem dan ketidakwajaran ini menunjukkan perlunya penanganan data lebih lanjut, yang akan dilakukan pada tahap data preparation.

#### **Distribusi Kelas pada Fitur Target**
Sebelum menganalisis lebih lanjut, penting untuk memahami proporsi masing-masing kategori untuk mengetahui apakah data seimbang atau tidak khususnya pada Fitur Targetnya. Hal ini akan membantu menentukan metode yang tepat pada tahap modeling, khususnya dalam kasus klasifikasi.

![image](https://github.com/user-attachments/assets/4dbb91ff-25cb-4c73-9249-e8e21177aaea)

Distribusi kelas target pada variabel `loan_status` menunjukkan ketidakseimbangan yang cukup signifikan. Kelas **0** (pinjaman tidak disetujui) mendominasi dataset, sementara kelas **1** (pinjaman disetujui) hanya mencakup sekitar **22,2%** dari total data. Ketimpangan distribusi ini perlu menjadi perhatian karena dapat memengaruhi kinerja model prediksi, terutama dalam hal akurasi terhadap kelas minoritas.

Oleh karena itu, pada tahap pelatihan model nantinya, akan dilakukan penyesuaian melalui metode **SMOTE (Synthetic Minority Oversampling Technique)**. Teknik ini digunakan untuk meningkatkan representasi kelas minoritas dengan cara membuat sampel sintetis, sehingga model dapat belajar secara lebih seimbang dari kedua kelas yang ada.

#### **Distribusi Fitur Numerik**
Selanjutnya, dilakukan visualisasi untuk memahami pola sebaran setiap variabel. Hal ini membantu mengidentifikasi apakah data berdistribusi normal, mencurigai adanya outlier, serta menentukan kebutuhan transformasi data di tahap selanjutnya.

![image](https://github.com/user-attachments/assets/12d991a1-5fe8-4a2a-966e-d5dc12a51a98)


## Interpretasi Distribusi Data Numerik

Berdasarkan grafik histogram di atas, didapatkan kesimpulan untuk masing-masing variabel, antara lain:

| Variabel                        | Pola Distribusi & Nilai Dominan                 | Kesimpulan                                                                |
|----------------------------------|-------------------------------------------------|--------------------------------------------------------------------------------------|
| **person_age**                   | Mayoritas usia 20–30 tahun, menurun tajam di atas 30 | Didominasi usia muda, pemohon kebanyakan di awal karir.                              |
| **person_income**                | Mayoritas < 100.000, outlier sangat tinggi       | Pendapatan pemohon umumnya rendah-menengah, beberapa sangat tinggi (outlier).        |
| **person_emp_exp**               | Mayoritas 0–5 tahun, menurun setelahnya          | Pengalaman kerja rendah, sesuai dengan usia muda pemohon.                            |
| **loan_amnt**                    | Puncak di 5.000–10.000, menurun setelah 15.000   | Pinjaman paling banyak diajukan dalam nominal kecil-menengah.                        |
| **loan_int_rate**                | Puncak di 10–13%, simetris, sedikit skew kanan   | Mayoritas pinjaman dengan bunga standar, sedikit yang sangat tinggi.                 |
| **loan_percent_income**          | Mayoritas < 0,2 (20%), menurun setelahnya        | Pinjaman umumnya proporsional kecil terhadap pendapatan.                             |
| **cb_person_cred_hist_length**   | Mayoritas 2–5 tahun, menurun setelahnya          | Riwayat kredit pendek, konsisten dengan usia & pengalaman kerja muda.                |
| **credit_score**                 | Rata-rata 600–700, distribusi normal             | Skor kredit pemohon mayoritas di level menengah, risiko kredit moderat.              |

#### **Melihat Seluruh Kategori pada Tiap Fitur Kategorik**
Selanjutnya, dilakukan identifikasi jumlah kategori unik pada setiap fitur kategorik. Langkah ini bertujuan untuk memahami variasi nilai yang mungkin memengaruhi proses encoding, serta mendeteksi kemungkinan nilai yang tidak konsisten atau redundan sebelum tahap data preparation.


| Fitur                               | Kategori (Jumlah)                                                                                                         | Tipe    | Alasan                                                                  |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------- | ----------------------------------------------------------------------- |
| **person\_gender**                     | Male (24.841), Female (20.159)                                                                                            | Nominal | Tidak ada urutan; kategori hanya identitas gender tanpa tingkatan.      |
| **person\_education**                  | Bachelor (13.399), Associate (12.028), High School (11.972), Master (6.980), Doctorate (621)                              | Ordinal | Pendidikan memiliki tingkatan urut dari rendah ke tinggi.               |
| **person\_home\_ownership**            | RENT (23.443), MORTGAGE (18.489), OWN (2.951), OTHER (117)                                                                | Nominal | Jenis kepemilikan rumah tidak memiliki urutan yang logis atau hierarki. |
| **loan\_intent**                       | EDUCATION (9.153), MEDICAL (8.548), VENTURE (7.819), PERSONAL (7.552), DEBTCONSOLIDATION (7.145), HOMEIMPROVEMENT (4.783) | Nominal | Tujuan pinjaman tidak memiliki tingkatan atau urutan tertentu.          |
| **previous\_loan\_defaults\_on\_file** | Yes (22.858), No (22.142)                                                                                                 | Nominal | Kategori biner tanpa urutan atau tingkatan.                             |


#### **Visualisasi Distribusi Tiap Fitur kategorik**
Setelah mengetahui jumlah kategori unik, dilakukan visualisasi untuk setiap variabel kategorik guna melihat distribusi frekuensinya dengan lebih jelas. Visualisasi ini membantu mengidentifikasi ketidakseimbangan antar kategori, pola dominasi kelas tertentu, serta potensi perlunya pengelompokan ulang kategori pada tahap persiapan data.

![image](https://github.com/user-attachments/assets/a6b1848e-8669-41d0-a931-f1e735ec7e13)


Berdasarkan visualisasi di atas, didapatkan beberapa informasi penting antara lain:

* Mayoritas pemohon adalah **pria**, menunjukkan dominasi gender laki-laki dalam pengajuan pinjaman.
* Tingkat pendidikan terbanyak adalah **Bachelor**, disusul oleh **Associate** dan **High School**, menunjukkan bahwa mayoritas pemohon berasal dari kelompok berpendidikan menengah hingga sarjana.
* Sebagian besar pemohon **menyewa rumah (RENT)** atau memiliki **rumah dengan hipotek (MORTGAGE)**, sedangkan yang **sepenuhnya memiliki rumah (OWN)** sangat sedikit.
* Tujuan pinjaman paling umum adalah untuk **EDUCATION**, kemudian **MEDICAL**, **PERSONAL**, dan **VENTURE**. Tujuan **HOMEIMPROVEMENT** paling jarang muncul.
* Riwayat gagal bayar tersebar **relatif merata** antara yang pernah gagal bayar (YES) dan yang tidak (NO), mengindikasikan bahwa risiko kredit dalam populasi ini cukup tinggi dan seimbang.

#### **Rata-rata Fitur Numerik Berdasarkan Kategori Target loan_status**
Selanjutnya, dilakukan analisis rata-rata variabel numerik berdasarkan kategori target **loan\_status**. Tujuannya adalah untuk melihat perbedaan karakteristik numerik antara pemohon yang gagal bayar dan yang melunasi pinjaman, sehingga dapat memberikan indikasi awal variabel mana yang potensial berpengaruh terhadap kelayakan kredit.

| loan_status | person_age | person_income | person_emp_exp | loan_amnt | loan_int_rate | loan_percent_income | cb_person_cred_hist_length | credit_score |
|-------------|------------|---------------|----------------|-----------|---------------|---------------------|----------------------------|--------------|
| 0           | 27.83      | 86157.04      | 5.48           | 9219.58   | 10.48         | 0.12                | 5.90                       | 632.81       |
| 1           | 27.52      | 59886.10      | 5.18           | 10855.69  | 12.86         | 0.20                | 5.76                       | 631.89       |


Berdasarkan tabel di atas didapatkan beberapa informasi penting, antara lain:
- Pemohon dengan status pinjaman lancar (0) cenderung memiliki pendapatan rata-rata yang lebih tinggi (86.157 vs 59.886) dan bunga pinjaman lebih rendah (10,48% vs 12,86%) dibandingkan dengan yang gagal bayar (1).
- Rata-rata jumlah pinjaman lebih besar pada kelompok gagal bayar (10.855 vs 9.220), menunjukkan potensi risiko yang lebih tinggi terkait besarnya pinjaman.
- Persentase pendapatan yang digunakan untuk membayar pinjaman juga lebih tinggi pada pemohon gagal bayar (20% vs 12%), yang bisa menjadi indikator tekanan finansial.
- Skor kredit hampir sama, namun sedikit lebih rendah pada kelompok gagal bayar.
- Faktor usia, pengalaman kerja, dan panjang riwayat kredit relatif mirip di kedua kelompok.
---


## **IV. Data Preparation**
Pada tahap **data preparation**, dilakukan serangkaian proses pembersihan dan scaling data agar siap digunakan dalam analisis dan pemodelan. Proses ini meliputi deteksi data duplikat dan missing values, deteksi dan pengelolaan outlier, encoding variabel kategorik, scaling data numerik, hingga splitting data dan teknik Smote. Tahap ini sangat penting untuk memastikan kualitas data yang optimal sehingga model yang dibangun dapat menghasilkan prediksi yang akurat dan handal.

#### IV.a. Periksa Data Duplikat 
Tahap ini dilakukan untuk memastikan dataset tidak memiliki data yang duplikat di dalamnya, agar tidak mengganggu proses analisis. Berdasarkan pemeriksaan didapatkan hasil untuk data duplikat sebagai berikut:
- Jumlah data duplikat: 0

Dapat disimpulkan bahwa tidak terdapat data duplikat, sehingga analisis dilanjutkan untuk memeriksa Missing Value.

#### IV.b. Periksa _Missing Value_
Pemeriksaan ini dilakukan untuk melihat apakah terdapat data yang hilang di dalamnya. Jika terdapat data yang hilang maka perlu penanganan lebih lanjut pada dataset ini. Berdasarkan pemeriksaan didapatkan hasil untuk _Missing Value_ sebagai berikut:

| Variabel                           | Jumlah Missing |
| ---------------------------------- | -------------- |
| person\_age                        | 0              |
| person\_gender                     | 0              |
| person\_education                  | 0              |
| person\_income                     | 0              |
| person\_emp\_exp                   | 0              |
| person\_home\_ownership            | 0              |
| loan\_amnt                         | 0              |
| loan\_intent                       | 0              |
| loan\_int\_rate                    | 0              |
| loan\_percent\_income              | 0              |
| cb\_person\_cred\_hist\_length     | 0              |
| credit\_score                      | 0              |
| previous\_loan\_defaults\_on\_file | 0              |
| **loan\_status**                   | 0              |
 

Berdasarkan pemeriksaan yang dilakukan, didapatkan hasil bahwa dataset bersih dari nilai kosong (_missing values_), sehingga analisis dapat dilanjutkan.

#### IV.c. Periksa Outlier Data
Sebelum membangun model machine learning, penting untuk memeriksa keberadaan outlier dalam data numerik. Outlier dapat memengaruhi distribusi data dan mengganggu kinerja model, terutama pada algoritma yang sensitif terhadap nilai ekstrem, sehingga pemeriksaan outlier ini penting untuk dilakukan.

Setelah dilakukan pemeriksaan Outlier menggunakan metode **IQR (Interquartile Range)**, yaitu dengan batas bawah (Q1 - 1.5 × IQR) dan batas atas (Q3 + 1.5 × IQR). Jumlah outlier yang ditemukan untuk masing-masing Fitur ditampilkan pada Tabel berikut:

| Fitur                       | Jumlah Outlier |
| ------------------------------ | -------------- |
| person\_age                    | 2.188          |
| person\_income                 | 2.218          |
| person\_emp\_exp               | 1.724          |
| loan\_amnt                     | 2.348          |
| loan\_int\_rate                | 124            |
| loan\_percent\_income          | 744            |
| cb\_person\_cred\_hist\_length | 1.366          |
| credit\_score                  | 467            |

Untuk menjaga integritas data namun tetap mengurangi pengaruh ekstrem, **penanganan outlier dilakukan menggunakan metode Winsorizing**. Metode ini membatasi nilai ekstrem dengan menggantinya ke nilai ambang batas tertentu, dalam hal ini, nilai-nilai yang berada di luar batas IQR (outlier) disesuaikan ke nilai Q1 atau Q3 sesuai arah outlier-nya.

Visualisasi data setelah dilakukan Winsorizing ditunjukkan pada boxplot berikut:

![image](https://github.com/user-attachments/assets/d1f5bf07-da97-40f6-9a01-96940456c1a7)

Dengan penerapan metode ini, distribusi data menjadi lebih representatif dan tidak terlalu dipengaruhi oleh nilai-nilai ekstrem, sehingga analisis berikutnya dapat dilakukan dengan hasil yang lebih reliabel.


### IV.d. Data Encoding
Setelah proses pembersihan dan penanganan outlier, tahap berikutnya adalah melakukan **data encoding** pada variabel kategorik. Encoding ini diperlukan agar data kategorik dapat diubah menjadi format numerik yang dapat diproses oleh algoritma machine learning, seperti menggunakan metode one-hot encoding atau Ordinal encoding sesuai kebutuhan model. Dalam dataset ini, dilakukan dua jenis encoding:
1. **Ordinal Encoding** untuk kolom dengan tingkat atau urutan.
2. **One-Hot Encoding** untuk kolom kategorikal nominal tanpa urutan.

Berikut adalah langkah-langkah yang dilakukan:
* Fitur **`person_education`** dikategorikan sebagai ordinal karena memiliki tingkatan pendidikan. Urutan tingkat pendidikan didefinisikan sebagai berikut:
  ```python
  education_order = ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']
  ```

  Kemudian, dilakukan encoding menggunakan `OrdinalEncoder`:

  ```python
  ord_enc = OrdinalEncoder(categories=[education_order])
  df_encode['person_education_enc'] = ord_enc.fit_transform(df_encode[['person_education']])
  ```

* Fitur kategorikal nominal seperti `person_gender`, `person_home_ownership`, `loan_intent`, dan `previous_loan_defaults_on_file` diencoding menggunakan **One-Hot Encoding** dengan `drop_first=True` untuk menghindari dummy variable trap:

  ```python
  df_encode = pd.get_dummies(df_encode, columns=nominal_cols, drop_first=True)
  ```

* Setelah proses encoding selesai, kolom asli dari fitur ordinal `person_education` dihapus:

  ```python
  df_encode.drop(columns=['person_education'], inplace=True)
  ```

Hasil encoding menghasilkan dataset baru `df_encode` yang siap diproses lebih lanjut.

### IV.e. Feature Scaling
Untuk memastikan bahwa semua fitur numerik memiliki skala yang sebanding, terutama agar model tidak bias terhadap fitur dengan rentang nilai yang besar, dilakukan **standardisasi (Standard Scaling)** menggunakan `StandardScaler` dari scikit-learn.

Kolom numerik kontinu yang dilakukan scaling meliputi:
* `person_age`
* `person_income`
* `person_emp_exp`
* `loan_amnt`
* `loan_int_rate`
* `loan_percent_income`
* `cb_person_cred_hist_length`
* `credit_score`

Langkah-langkahnya adalah sebagai berikut:
```python
scaler = StandardScaler()
df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])
```
Setelah dilakukan scaling, semua nilai pada kolom tersebut memiliki distribusi dengan **rata-rata 0** dan **standar deviasi 1**, yang membantu meningkatkan stabilitas dan performa dari model machine learning yang akan dibangun.

### IV.c. Feature Selection
Setelah melakukan encoding dan scaling pada data, langkah berikutnya adalah melakukan **seleksi fitur (feature selection)** untuk meningkatkan efisiensi dan kinerja model. Proses seleksi fitur dilakukan menggunakan algoritma **Random Forest Classifier**, karena robust terhadap multikolinearitas dan mampu memberikan estimasi tingkat kepentingan (importance) dari setiap fitur.

Langkah-langkah yang dilakukan:
1. Data input (`X`) diambil dari `df_scaled` dengan menghapus kolom target `loan_status`.
2. Data target (`y`) adalah kolom `loan_status`.
3. Model Random Forest dilatih untuk menilai pentingnya setiap fitur.
4. Hasil feature importance divisualisasikan dalam bentuk bar chart horizontal untuk interpretasi yang lebih jelas.

![image](https://github.com/user-attachments/assets/b0d5cde0-1a0c-403b-a2a7-17767e7f73f1)

Berikut adalah hasil fitur-fitur yang memiliki tingkat kepentingan di atas _threshold_ 0.01:
| Feature                         | Importance  |
|--------------------------------|-------------|
| previous_loan_defaults_on_file_Yes | 0.229956    |
| loan_percent_income             | 0.163062    |
| loan_int_rate                  | 0.157834    |
| person_income                  | 0.123191    |
| loan_amnt                     | 0.058246    |
| person_home_ownership_RENT     | 0.056578    |
| credit_score                  | 0.055034    |
| person_age                    | 0.028899    |
| person_emp_exp                | 0.026750    |
| cb_person_cred_hist_length    | 0.024243    |
| person_education_enc          | 0.017201    |
| person_home_ownership_OWN     | 0.010737    |

Fitur-fitur inilah yang digunakan untuk membangun model prediktif pada tahap selanjutnya.

### IV.f. Train-Test Split
Setelah fitur-fitur penting dipilih, dilakukan pemisahan data menjadi data latih (**training set**) dan data uji (**testing set**). Proses ini penting agar performa model dapat dievaluasi secara objektif terhadap data yang belum pernah dilihat.

* Proporsi pembagian data adalah **80% untuk pelatihan dan 20% untuk pengujian**.
* Pembagian dilakukan secara **stratified** untuk memastikan distribusi kelas target `loan_status` tetap terjaga.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y)
```

### IV.g. Penanganan Ketidakseimbangan Kelas (SMOTE)
Seperti yang telah dijelaskan sebelumnya, distribusi target `loan_status` sangat tidak seimbang, dengan kelas 1 (pinjaman disetujui) hanya sekitar 22,2% dari total data. Ketidakseimbangan ini dapat mengakibatkan model bias terhadap kelas mayoritas (pinjaman ditolak).

Untuk mengatasi hal ini, digunakan metode **SMOTE (Synthetic Minority Over-sampling Technique)**, yang bekerja dengan membuat contoh sintetis dari kelas minoritas berdasarkan tetangga terdekatnya.

Distribusi target sebelum dan sesudah penerapan SMOTE:
```
Original train target distribution:
0    28000
1     8000

After SMOTE train target distribution:
0    28000
1    28000
```

Dengan SMOTE, jumlah data dari kedua kelas pada data latih menjadi seimbang, yang diharapkan dapat meningkatkan kemampuan model dalam mengenali kelas minoritas.

---


## **V. Modeling**
Pada tahap ini, dilakukan pembangunan model machine learning untuk memprediksi status persetujuan pinjaman (`loan_status`). Dua algoritma yang dipilih untuk pemodelan adalah **Logistic Regression** dan **XGBoost Classifier**. Kedua algoritma tersebut dipilih karena memiliki karakteristik yang berbeda dan umum digunakan untuk masalah klasifikasi biner.

### 1. Logistic Regression

Logistic Regression merupakan model linear yang sederhana dan mudah diinterpretasikan. Model ini memodelkan probabilitas kelas target berdasarkan kombinasi linear dari fitur input.

**Tahapan dan parameter tuning:**

* Model diinisialisasi dengan `max_iter=1000` agar memastikan konvergensi.
* Parameter yang dituning melalui `GridSearchCV` dengan 5-fold cross-validation meliputi:

  * `C`: parameter regularisasi yang mengatur kekuatan penalti (0.01, 0.1, 1, 10).
  * `penalty`: jenis regularisasi yang digunakan, dipilih `l2` (Ridge).
  * `solver`: algoritma optimisasi, menggunakan `lbfgs` yang cocok untuk `l2` penalty dan dataset berukuran sedang.

Grid search ini bertujuan untuk menemukan kombinasi parameter terbaik yang memaksimalkan akurasi model pada data latih yang sudah di-balance dengan SMOTE.

**Kelebihan Logistic Regression:**

* Model sederhana dan cepat dilatih.
* Interpretasi koefisien mudah untuk memahami pengaruh fitur.
* Kurang rentan terhadap overfitting pada data yang bersih dan berukuran cukup besar.

**Kekurangan Logistic Regression:**

* Hanya mampu memodelkan hubungan linear antara fitur dan target.
* Kurang optimal jika hubungan antara fitur dan target bersifat kompleks atau non-linear.

---

### 2. XGBoost Classifier

XGBoost adalah algoritma boosting berbasis pohon keputusan yang kuat dan sering memberikan performa tinggi pada banyak dataset klasifikasi.

**Tahapan dan parameter tuning:**

* Model diinisialisasi dengan parameter default, dan `use_label_encoder=False` serta `eval_metric='logloss'` untuk menghindari warning dan menggunakan log-loss sebagai fungsi evaluasi.
* Parameter yang dituning dengan `GridSearchCV` menggunakan 5-fold cross-validation meliputi:

  * `n_estimators`: jumlah pohon yang dibangun (100, 200).
  * `max_depth`: kedalaman maksimum pohon (3, 6).
  * `learning_rate`: kecepatan pembelajaran (0.01, 0.1).
  * `subsample`: proporsi data yang diambil tiap iterasi boosting (0.7, 1).

Proses tuning ini bertujuan menemukan konfigurasi terbaik untuk menghindari overfitting sekaligus meningkatkan generalisasi model.

**Kelebihan XGBoost:**

* Mampu menangkap hubungan non-linear dan interaksi fitur.
* Memiliki mekanisme regularisasi yang kuat untuk mencegah overfitting.
* Umumnya memberikan performa akurasi tinggi pada berbagai masalah klasifikasi.

**Kekurangan XGBoost:**

* Proses training lebih lambat dibandingkan model linear.
* Interpretasi model lebih kompleks dibandingkan Logistic Regression.
* Memerlukan tuning parameter lebih intensif agar performa optimal.
---

### Pemilihan Model Terbaik
Pemilihan model terbaik didasarkan pada keseimbangan antara kemampuan prediksi yang akurat dan generalisasi model terhadap data baru. Misalnya, jika XGBoost menunjukkan performa metrik evaluasi yang lebih unggul dibanding Logistic Regression, terutama dalam menangani data yang tidak seimbang dan fitur yang kompleks, maka XGBoost dipilih sebagai model terbaik. Namun, apabila Logistic Regression memberikan hasil yang cukup baik dengan kelebihan interpretabilitas yang tinggi dan proses pelatihan yang lebih sederhana, maka model ini bisa menjadi pilihan terutama untuk kebutuhan penjelasan hasil kepada stakeholder.

---


## VI. Evaluation

Pada tahap evaluasi ini, digunakan beberapa metrik untuk mengukur performa model machine learning yang telah dibangun, yaitu:

* **Accuracy**
  Mengukur proporsi prediksi yang benar terhadap seluruh data.
  Formula:

  $$
  \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
  $$

  Namun, accuracy tidak selalu ideal untuk dataset tidak seimbang karena bisa bias terhadap kelas mayoritas.

* **Precision, Recall, dan F1-score**
  Digunakan untuk melihat performa model lebih detail pada tiap kelas, terutama kelas minoritas (pinjaman disetujui).

  * Precision: proporsi prediksi positif yang benar dari seluruh prediksi positif.
  * Recall: proporsi data positif yang benar terdeteksi oleh model.
  * F1-score: harmonic mean dari precision dan recall, mengombinasikan keduanya menjadi satu metrik.

* **ROC-AUC (Area Under the ROC Curve)**
  Mengukur kemampuan model membedakan kelas positif dan negatif secara keseluruhan dengan mengombinasikan tingkat True Positive Rate (Recall) dan False Positive Rate pada berbagai threshold.
  Nilai ROC-AUC berkisar dari 0.5 (acak) sampai 1 (sempurna).

* **Confusion Matrix**
Selain metrik evaluasi seperti akurasi, precision, recall, F1-score, dan AUC-ROC, **confusion matrix** juga digunakan untuk memberikan gambaran detail tentang performa model. Confusion matrix menampilkan jumlah prediksi yang benar dan salah dalam setiap kelas, yaitu:

   - **True Positive (TP)**: prediksi positif yang benar
   - **True Negative (TN)**: prediksi negatif yang benar
   - **False Positive (FP)**: prediksi positif yang salah
   - **False Negative (FN)**: prediksi negatif yang salah

   Dengan melihat confusion matrix, jenis kesalahan yang sering terjadi dapat lebih dipahami, sehingga dapat mengambil langkah perbaikan yang lebih tepat sesuai tujuan bisnis atau analisis.


### Hasil Evaluasi Model

Evaluasi dilakukan pada data testing dengan hasil sebagai berikut:

| Model               | Accuracy | Precision (Kelas 1) | Recall (Kelas 1) | F1-Score (Kelas 1) | ROC-AUC |
| ------------------- | -------- | ------------------- | ---------------- | ------------------ | ------- |
| Logistic Regression | 0.858    | 0.62                | 0.91             | 0.74               | 0.9534  |
| XGBoost             | 0.920    | 0.82                | 0.82             | 0.82               | 0.9712  |

* Logistic Regression memiliki recall yang tinggi pada kelas minoritas (0.91), artinya model cukup baik dalam menangkap kasus pinjaman disetujui, namun presisinya rendah (0.62) sehingga banyak prediksi positif yang salah. Hal ini menyebabkan F1-score menjadi moderat (0.74).
* XGBoost menunjukkan keseimbangan yang lebih baik antara precision dan recall (keduanya 0.82), sehingga F1-score juga lebih tinggi (0.82). Accuracy dan ROC-AUC-nya juga lebih unggul dibanding Logistic Regression.

### Visualisasi Evaluasi

* **ROC Curve**: Grafik menunjukkan bahwa XGBoost memiliki kurva ROC yang lebih tinggi dibanding Logistic Regression, mengindikasikan kemampuan pemisahan kelas yang lebih baik. Skor AUC untuk XGBoost juga lebih baik dibandingkan Logistic Regression.

   ![image](https://github.com/user-attachments/assets/372de6f0-d20e-46d9-8194-31401fdaca3e)


* **Confusion Matrix**: Visualisasi matriks menunjukkan bahwa XGBoost memiliki distribusi prediksi yang lebih akurat dengan jumlah true positive dan true negative yang lebih tinggi dibandingkan Logistic Regression. Hal ini mengindikasikan bahwa XGBoost mampu mengklasifikasikan data dengan kesalahan yang lebih sedikit, sehingga performa model ini lebih baik dalam memprediksi status pinjaman secara tepat.

   ![image](https://github.com/user-attachments/assets/918ac774-726e-4f94-a70d-26c5966212bc)
   ![image](https://github.com/user-attachments/assets/f6c26811-2184-45de-8d31-5281ebc77760)


* **Train dan Test Accuracy**: XGBoost memiliki akurasi pelatihan yang lebih tinggi (0.9443) dibandingkan Logistic Regression (0.8547), dan juga performa testing yang lebih baik, namun tidak menunjukkan tanda overfitting yang signifikan.

   ![image](https://github.com/user-attachments/assets/c2fe93f0-dd12-4ee7-96c9-d1c1fb35cbca)


## Kesimpulan
### **Pemilihan Model Terbaik**

Berdasarkan hasil seluruh evaluasi di atas, **XGBoost dipilih sebagai model terbaik** karena memiliki performa yang lebih unggul secara keseluruhan, baik dari sisi accuracy, precision, recall, F1-score, maupun ROC-AUC hingga Confussion Matrix. Model ini mampu menangani ketidakseimbangan kelas dengan baik setelah penerapan SMOTE dan hyperparameter tuning, sehingga lebih dapat diandalkan untuk memprediksi status pinjaman.

---

### **Menghubungkan Evaluasi dengan Business Understanding**
#### Jawaban Pernyataan Masalah 1

**Fitur-fitur apa yang paling berpengaruh dalam menentukan kelayakan persetujuan pinjaman bagi seorang pemohon?**

Berdasarkan analisis feature importance, fitur yang paling berpengaruh dalam menentukan kelayakan persetujuan pinjaman adalah:

* **Riwayat gagal bayar sebelumnya (previous\_loan\_defaults\_on\_file\_Yes)** dengan nilai importance tertinggi, menunjukkan bahwa pemohon yang pernah gagal bayar memiliki risiko lebih tinggi untuk ditolak.
* **Persentase penghasilan yang dialokasikan untuk pinjaman (loan\_percent\_income)** dan **tingkat bunga pinjaman (loan\_int\_rate)** juga berkontribusi besar, karena menunjukkan beban keuangan pemohon.
* Fitur finansial lain seperti **penghasilan pribadi (person\_income)**, **jumlah pinjaman (loan\_amnt)**, serta **skor kredit (credit\_score)** memberikan sinyal penting mengenai kemampuan bayar.
* Faktor demografi dan latar belakang seperti **umur pemohon (person\_age)**, **lama pengalaman kerja (person\_emp\_exp)**, dan **lama riwayat kredit (cb\_person\_cred\_hist\_length)** juga mempengaruhi keputusan.
* Status kepemilikan rumah (RENT dan OWN) dan tingkat pendidikan turut memberikan pengaruh, meskipun lebih kecil.

---

#### Jawaban Pernyataan Masalah 2

**Bagaimana membangun model prediksi klasifikasi biner (approve/tolak) yang akurat berdasarkan fitur terpilih?**

Model klasifikasi dibangun dengan menggunakan fitur-fitur terpilih berdasarkan nilai importance di atas threshold. Dua algoritma yang diuji adalah Logistic Regression dan XGBoost, dengan proses tuning hyperparameter dan penanganan imbalance data menggunakan SMOTE. Hasil evaluasi menunjukkan bahwa model **XGBoost** memberikan performa terbaik dengan akurasi, precision, recall, F1-score, dan ROC-AUC yang lebih unggul dibandingkan Logistic Regression. Oleh karena itu, XGBoost dipilih sebagai model utama untuk memprediksi status persetujuan pinjaman secara akurat.

---
#### Jawaban Pernyataan Masalah 3

**Bagaimana memastikan model dapat melakukan generalisasi yang baik terhadap data baru yang belum pernah dilihat sebelumnya?**

Model XGBoost dipastikan memiliki kemampuan generalisasi yang baik melalui proses evaluasi yang komprehensif dan berlapis, yaitu:

* **Cross-validation 5-fold** digunakan selama tuning hyperparameter untuk menguji kestabilan performa model pada beberapa subset data yang berbeda, sehingga meminimalisasi risiko overfitting pada data pelatihan.
* **Penanganan ketidakseimbangan data dengan SMOTE** membantu model belajar dengan lebih seimbang, khususnya pada kelas minoritas, sehingga prediksi pada data baru menjadi lebih akurat.
* **Evaluasi performa pada data testing yang terpisah** menunjukkan bahwa nilai akurasi, precision, recall, F1-score, dan ROC-AUC XGBoost tetap tinggi dan seimbang dengan hasil pada data training.
* Tidak terdapat penurunan performa signifikan antara data training (akurasi 0.9443) dan data testing (akurasi 0.920), menandakan model tidak mengalami overfitting yang berarti dan dapat mengeneralisasi pola data dengan baik.

Dengan proses evaluasi yang matang ini, dapat disimpulkan bahwa model XGBoost memiliki kemampuan generalisasi yang kuat dan dapat diandalkan untuk memprediksi status persetujuan pinjaman pada data baru secara konsisten dan akurat.

---
