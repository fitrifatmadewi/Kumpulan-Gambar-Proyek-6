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

#### **Periksa Data Duplikat dan Missing Value**
Berdasarkan pemeriksaan didapatkan hasil untuk data duplikat dan _Missing Value_ sebagai berikut:
- Jumlah data duplikat: 0

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
 

Berdasarkan pemeriksaan yang dilakukan, didapatkan hasil bahwa dataset bersih dari duplikat dan tidak memiliki nilai kosong (_missing values_), sehingga analisis dapat dilanjutkan.

#### **Periksa Outlier Data**
Outlier telah diidentifikasi menggunakan metode **IQR (Interquartile Range)**, yaitu dengan batas bawah (Q1 - 1.5 × IQR) dan batas atas (Q3 + 1.5 × IQR). Jumlah outlier yang ditemukan untuk masing-masing Fitur ditampilkan pada Tabel berikut:

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

Untuk menjaga integritas data namun tetap mengurangi pengaruh ekstrem, **penanganan outlier dilakukan menggunakan metode Winsorizing**. Metode ini membatasi nilai ekstrem dengan menggantinya ke nilai ambang batas tertentu — dalam hal ini, nilai-nilai yang berada di luar batas IQR (outlier) disesuaikan ke nilai Q1 atau Q3 sesuai arah outlier-nya.

Visualisasi data setelah dilakukan Winsorizing ditunjukkan pada boxplot berikut:

![image](https://github.com/user-attachments/assets/077bcc31-e38e-451f-a8cd-408fca0ef022)


Dengan penerapan metode ini, distribusi data menjadi lebih representatif dan tidak terlalu dipengaruhi oleh nilai-nilai ekstrem, sehingga analisis berikutnya dapat dilakukan dengan hasil yang lebih reliabel.

#### **Statistika Deskriptif untuk Fitur Numerik**

| Statistik    | person\_age | person\_income | person\_emp\_exp | loan\_amnt | loan\_int\_rate | loan\_percent\_income | cb\_person\_cred\_hist\_length | credit\_score | loan\_status |
| ------------ | ----------- | -------------- | ---------------- | ---------- | --------------- | --------------------- | ------------------------------ | ------------- | ------------ |
| count        | 45000.0     | 45000.0        | 45000.0          | 45000.0    | 45000.0         | 45000.0               | 45000.0                        | 45000.0       | 45000.0      |
| mean         | 27.44       | 75677.4        | 5.17             | 9411.04    | 11.01           | 0.14                  | 5.78                           | 632.81        | 0.22         |
| std          | 4.93        | 38071.78       | 5.14             | 5832.95    | 2.98            | 0.08                  | 3.58                           | 49.80         | 0.42         |
| min          | 20.0        | 8000.0         | 0.0              | 500.0      | 5.42            | 0.0                   | 2.0                            | 497.5         | 0.0          |
| 25%          | 24.0        | 47204.0        | 1.0              | 5000.0     | 8.59            | 0.07                  | 3.0                            | 601.0         | 0.0          |
| 50% (median) | 26.0        | 67048.0        | 4.0              | 8000.0     | 11.01           | 0.12                  | 4.0                            | 640.0         | 0.0          |
| 75%          | 30.0        | 95789.25       | 8.0              | 12237.25   | 12.99           | 0.19                  | 8.0                            | 670.0         | 0.0          |
| max          | 39.0        | 168667.13      | 18.5             | 23093.13   | 19.59           | 0.37                  | 15.5                           | 773.5         | 1.0          |

Berdasarkan statistika deskriptif di atas didapatkan beberapa informasi penting, antara lain:
- Usia pemohon berkisar antara 20–39 tahun dengan rata-rata 27 tahun.
- Pendapatan pemohon sangat bervariasi, tetapi sebagian besar berada di bawah 100.000.
- Tingkat bunga pinjaman rata-rata adalah 11%, dengan rentang antara 5,4–19,6%.
- Sebagian besar rasio pinjaman terhadap pendapatan berada di bawah 20%, menunjukkan nilai pinjaman relatif kecil terhadap kemampuan bayar.
- Skor kredit pemohon umumnya berada di level menengah (rata-rata 633).

#### **Distribusi Kelas pada Fitur Target**
![image](https://github.com/user-attachments/assets/4dbb91ff-25cb-4c73-9249-e8e21177aaea)

Distribusi kelas target pada variabel `loan_status` menunjukkan ketidakseimbangan yang cukup signifikan. Kelas **0** (pinjaman tidak disetujui) mendominasi dataset, sementara kelas **1** (pinjaman disetujui) hanya mencakup sekitar **22,2%** dari total data. Ketimpangan distribusi ini perlu menjadi perhatian karena dapat memengaruhi kinerja model prediksi, terutama dalam hal akurasi terhadap kelas minoritas.

Oleh karena itu, pada tahap pelatihan model nantinya, akan dilakukan penyesuaian melalui metode **SMOTE (Synthetic Minority Oversampling Technique)**. Teknik ini digunakan untuk meningkatkan representasi kelas minoritas dengan cara membuat sampel sintetis, sehingga model dapat belajar secara lebih seimbang dari kedua kelas yang ada.

#### **Distribusi Fitur Numerik**
![image](https://github.com/user-attachments/assets/8e245f48-e103-4b4f-8fe1-3c2ff0047d28)

Berdasarkan grafik di atas, didapatkan informasi sebagai berikut:
**1. Distribusi person_age**  
Mayoritas usia pemohon berada di rentang 22–28 tahun, dengan puncak pada usia awal 20-an. Setelah itu, jumlah pemohon menurun seiring bertambahnya usia, menunjukkan bahwa kebanyakan pemohon kredit adalah generasi muda atau awal karir.

**2. Distribusi person_income**  
Distribusi pendapatan cenderung miring ke kanan (right-skewed), dengan sebagian besar pemohon memiliki pendapatan di bawah 100.000. Ada sedikit lonjakan pada pendapatan yang sangat tinggi, namun jumlahnya jauh lebih sedikit.

**3. Distribusi person_emp_exp**  
Pengalaman kerja pemohon didominasi oleh mereka dengan pengalaman kerja 0–2 tahun, lalu menurun drastis seiring bertambahnya tahun pengalaman. Ini konsisten dengan distribusi usia yang didominasi oleh usia muda.

**4. Distribusi loan_amnt**  
Jumlah pinjaman yang diajukan bervariasi, namun terdapat beberapa kelompok nominal pinjaman yang sering diajukan (terlihat dari beberapa puncak pada histogram). Mayoritas pinjaman berada di bawah 10.000.

**5. Distribusi loan_int_rate**  
Sebagian besar tingkat bunga pinjaman berkisar antara 10–13%, dengan puncak tajam di sekitar 12%. Setelah itu, distribusi menurun tajam, menandakan mayoritas pinjaman diberikan dengan bunga standar di kisaran tersebut.

**6. Distribusi loan_percent_income**  
Rasio jumlah pinjaman terhadap pendapatan pemohon umumnya berada di bawah 0,2 (20%), dengan sebagian besar pemohon mengajukan pinjaman yang relatif kecil dibandingkan pendapatan mereka.

**7. Distribusi cb_person_cred_hist_length**  
Panjang riwayat kredit mayoritas pemohon berada di 2–4 tahun, lalu menurun drastis. Hal ini juga sejalan dengan usia dan pengalaman kerja yang relatif muda.

**8. Distribusi credit_score**  
Skor kredit pemohon membentuk distribusi mendekati normal dengan rata-rata di kisaran 650. Sebagian besar pemohon memiliki skor kredit antara 600–700, dengan sedikit yang memiliki skor sangat rendah atau sangat tinggi.

#### **Melihat Seluruh Kategori pada Tiap Fitur Kategorik**

| Fitur                               | Kategori (Jumlah)                                                                                                         | Tipe    | Alasan                                                                  |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------- | ----------------------------------------------------------------------- |
| **person\_gender**                     | Male (24.841), Female (20.159)                                                                                            | Nominal | Tidak ada urutan; kategori hanya identitas gender tanpa tingkatan.      |
| **person\_education**                  | Bachelor (13.399), Associate (12.028), High School (11.972), Master (6.980), Doctorate (621)                              | Ordinal | Pendidikan memiliki tingkatan urut dari rendah ke tinggi.               |
| **person\_home\_ownership**            | RENT (23.443), MORTGAGE (18.489), OWN (2.951), OTHER (117)                                                                | Nominal | Jenis kepemilikan rumah tidak memiliki urutan yang logis atau hierarki. |
| **loan\_intent**                       | EDUCATION (9.153), MEDICAL (8.548), VENTURE (7.819), PERSONAL (7.552), DEBTCONSOLIDATION (7.145), HOMEIMPROVEMENT (4.783) | Nominal | Tujuan pinjaman tidak memiliki tingkatan atau urutan tertentu.          |
| **previous\_loan\_defaults\_on\_file** | Yes (22.858), No (22.142)                                                                                                 | Nominal | Kategori biner tanpa urutan atau tingkatan.                             |


#### **Visualisasi Distribusi Tiap Fitur kategorik**
![image](https://github.com/user-attachments/assets/4826d271-0e8e-4799-918c-eba10c808368)

Berdasarkan visualisasi di atas, didapatkan beberapa informasi penting antara lain:

* Mayoritas pemohon adalah **pria**, menunjukkan dominasi gender laki-laki dalam pengajuan pinjaman.
* Tingkat pendidikan terbanyak adalah **Bachelor**, disusul oleh **Associate** dan **High School**, menunjukkan bahwa mayoritas pemohon berasal dari kelompok berpendidikan menengah hingga sarjana.
* Sebagian besar pemohon **menyewa rumah (RENT)** atau memiliki **rumah dengan hipotek (MORTGAGE)**, sedangkan yang **sepenuhnya memiliki rumah (OWN)** sangat sedikit.
* Tujuan pinjaman paling umum adalah untuk **EDUCATION**, kemudian **MEDICAL**, **PERSONAL**, dan **VENTURE**. Tujuan **HOMEIMPROVEMENT** paling jarang muncul.
* Riwayat gagal bayar tersebar **relatif merata** antara yang pernah gagal bayar (YES) dan yang tidak (NO), mengindikasikan bahwa risiko kredit dalam populasi ini cukup tinggi dan seimbang.

#### **Rata-rata Fitur Numerik Berdasarkan Kategori Target loan_status**

| loan\_status  | person\_age | person\_income | person\_emp\_exp | loan\_amnt | loan\_int\_rate | loan\_percent\_income | cb\_person\_cred\_hist\_length | credit\_score |
| ------------- | ----------- | -------------- | ---------------- | ---------- | --------------- | --------------------- | ------------------------------ | ------------- |
| 0 (Ditolak)   | 27.51       | 80,747.53      | 5.24             | 9,076.48   | 10.48           | 0.12                  | 5.82                           | 633.01        |
| 1 (Disetujui) | 27.21       | 57,931.95      | 4.94             | 10,582.00  | 12.85           | 0.20                  | 5.67                           | 632.09        |

Berdasarkan tabel di atas didapatkan beberapa informasi penting, antara lain:
- Pemohon yang ditolak cenderung memiliki pendapatan lebih tinggi, pinjaman lebih kecil, dan bunga lebih rendah.
- Pemohon yang disetujui justru mengajukan pinjaman lebih besar dengan bunga lebih tinggi, dan rasio pinjaman terhadap pendapatan lebih tinggi.
- Hal ini bisa menandakan bahwa keputusan persetujuan lebih kompleks dan tidak semata-mata berdasarkan pendapatan, tapi juga pada profil risiko dan tujuan pinjaman.
---


## **IV. Data Preprocessing**

### IV.a. Data Encoding
Untuk mempersiapkan data agar dapat digunakan dalam model machine learning, diperlukan proses encoding terhadap fitur kategorikal. Dalam dataset ini, dilakukan dua jenis encoding:
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

### IV.b. Feature Scaling
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
![image](https://github.com/user-attachments/assets/4103ebd7-86e6-4e3e-89e9-02158074d222)

Berikut adalah hasil fitur-fitur yang memiliki tingkat kepentingan di atas _threshold_ 0.01:
```python
Selected Features: [
    'person_home_ownership_OWN', 
    'person_education_enc', 
    'cb_person_cred_hist_length', 
    'person_emp_exp', 
    'person_age', 
    'credit_score', 
    'person_home_ownership_RENT', 
    'loan_amnt', 
    'person_income', 
    'loan_int_rate', 
    'loan_percent_income', 
    'previous_loan_defaults_on_file_Yes'
]
```
Fitur-fitur inilah yang digunakan untuk membangun model prediktif pada tahap selanjutnya.

### IV.d. Train-Test Split
Setelah fitur-fitur penting dipilih, dilakukan pemisahan data menjadi data latih (**training set**) dan data uji (**testing set**). Proses ini penting agar performa model dapat dievaluasi secara objektif terhadap data yang belum pernah dilihat.

* Proporsi pembagian data adalah **80% untuk pelatihan dan 20% untuk pengujian**.
* Pembagian dilakukan secara **stratified** untuk memastikan distribusi kelas target `loan_status` tetap terjaga.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y)
```

### IV.e. Penanganan Ketidakseimbangan Kelas (SMOTE)
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


* **Confusion Matrix**: Visualisasi matriks kebingungan memperlihatkan distribusi prediksi benar dan salah untuk masing-masing model, memberikan gambaran rinci kesalahan klasifikasi.
![image](https://github.com/user-attachments/assets/918ac774-726e-4f94-a70d-26c5966212bc)
![image](https://github.com/user-attachments/assets/f6c26811-2184-45de-8d31-5281ebc77760)


* **Train dan Test Accuracy**: XGBoost memiliki akurasi pelatihan yang lebih tinggi (0.9443) dibandingkan Logistic Regression (0.8547), dan juga performa testing yang lebih baik, namun tidak menunjukkan tanda overfitting yang signifikan.
![image](https://github.com/user-attachments/assets/c2fe93f0-dd12-4ee7-96c9-d1c1fb35cbca)


### Kesimpulan

Berdasarkan hasil metrik evaluasi di atas, **XGBoost dipilih sebagai model terbaik** karena memiliki performa yang lebih unggul secara keseluruhan, baik dari sisi accuracy, precision, recall, F1-score, maupun ROC-AUC. Model ini mampu menangani ketidakseimbangan kelas dengan baik setelah penerapan SMOTE dan hyperparameter tuning, sehingga lebih dapat diandalkan untuk memprediksi status pinjaman.
