# Laporan Proyek Machine Learning - Muhammad Salwa Fairus Santoso
 
## 1. Domain Proyek
### Latar Belakang
Dalam industri perbankan, strategi pemasaran untuk produk seperti term deposit sering kali melibatkan pendekatan langsung kepada nasabah, seperti melalui telemarketing. Namun, pendekatan ini memiliki tantangan besar dalam hal efektivitas dan efisiensi biaya. Banyak sumber daya yang dikeluarkan untuk menjangkau calon pelanggan yang pada akhirnya tidak tertarik atau tidak memenuhi syarat untuk berlangganan produk tersebut.
 
Masalah ini berkaitan erat dengan customer adoption, yaitu seberapa besar kemungkinan nasabah untuk menerima dan menggunakan produk-produk perbankan yang ditawarkan. Dengan menganalisis informasi pelanggan yang tersedia, kita dapat membangun model prediktif untuk mengetahui nasabah mana yang berpotensi tinggi untuk melakukan subscribe terhadap term deposit.
 
 
##### Mengapa
- Biaya Operasional yang Sangat Besar
Tanpa model prediksi, bank harus menghubungi semua pelanggan tanpa mengetahui siapa yang benar-benar tertarik, yang menyebabkan biaya marketing sangat besar, terutama jika basis pelanggan mencapai jutaan.
  
- Inefisiensi dalam Strategi Pemasaran
Sumber daya (tenaga kerja, waktu, uang) dihabiskan untuk menghubungi pelanggan yang kemungkinan besar tidak akan berlangganan.Hal ini menurunkan Return on Investment (ROI) dalam aktivitas pemasaran.
 
##### Bagaimana
- Menggunakan Model Prediksi (Machine Learning)
Model seperti Random Forest digunakan untuk mengklasifikasikan pelanggan yang berpotensi tinggi untuk berlangganan.
 
- Fokus pada Efisiensi Biaya
Hanya pelanggan yang diprediksi akan berlangganan yang dihubungi, mengurangi biaya call center. Misal dari 1 juta pelanggan, hanya sekitar 260 ribu yang dihubungi, bukan semuanya.
 
 Referensi: [Penerapan Algoritma Machine Learning Untuk Memprediksi Term Deposit Nasabah Perbankan](https://journal.ittelkom-pwt.ac.id/index.php/ledger/article/view/1416) 
 
## 2. Business Understanding
Bank menghadapi tantangan dalam pemasaran produk term deposit karena pendekatan telemarketing yang tidak terarah menyebabkan biaya operasional tinggi dan efektivitas yang rendah. Tanpa model prediktif, semua nasabah dihubungi tanpa mengetahui siapa yang benar-benar tertarik.
 
Proyek ini bertujuan untuk mengembangkan model machine learning yang dapat memprediksi apakah seorang nasabah berpotensi berlangganan term deposit atau tidak. Dengan model ini, bank dapat menargetkan nasabah yang relevan, sehingga kampanye lebih efisien, hemat biaya, dan tepat sasaran.
### Problem Statements
- Bagaimana membuat model machine learning yang dapat memprediksi apakah seorang nasabah akan berlangganan term deposit atau tidak.?
- Model seperti apa yang memiliki akurasi paling baik?
- Bagaimana model ini dapat membantu pihak bank dalam meningkatkan efektivitas pemasaran produk term deposit?
 
 
### Goals
Menjelaskan tujuan dari pernyataan masalah:
- Membangun model prediktif untuk mengklasifikasikan nasabah ke dalam dua kategori: berpotensi berlangganan atau tidak dalam term deposit.
- Membandingkan beberapa model untuk menemukan recall terbaik dalam memprediksi pelanggan yang akan melakukan term deposit
- Pihak bank hanya perlu menghubungi nasabah yang diprediksi memiliki potensi tinggi untuk berlangganan, sehingga jumlah nasabah yang perlu dihubungi berkurang secara signifikan berkat bantuan model prediktif ini.
 
 
### Solution statements
- Menerapkan model klasifikasi Decision Tree, Random Forest dan Random Forest (Hyperparameter Tunimg) untuk memprediksi keputusan pelanggan.
- Melakukan evaluasi performa model menggunakan metrik seperti  precision.
- Memilih model terbaik berdasarkan precision tertinggi agar bank dapat mengurangi jumlah nasabah yang perlu dihubungi, sekaligus meningkatkan efisiensi biaya dan akurasi target pemasaran.
 
## 3. Data Understanding
Data Understanding merupakan tahap penting dalam proses analisis data karena berfungsi untuk memahami struktur, kualitas, dan karakteristik data yang akan digunakan dalam pemodelan. Dalam proyek ini, dataset yang digunakan terdiri dari 11.162 baris dan 17 kolom tanpa adanya nilai yang hilang maupun data duplikat, yang menunjukkan bahwa dataset ini telah cukup bersih dan siap untuk dianalisis lebih lanjut. Namun demikian, proses pemeriksaan lebih dalam menunjukkan adanya sejumlah outlier pada beberapa fitur numerik seperti balance, duration, campaign, pdays, dan previous. Misalnya, fitur pdays memiliki 2.750 nilai outlier, sementara balance memiliki 1.055 outlier yang mengindikasikan adanya nasabah dengan saldo yang tidak biasa. Deteksi outlier ini penting untuk memastikan bahwa model tidak bias terhadap nilai-nilai ekstrem yang dapat memengaruhi hasil prediksi. Dengan memahami struktur dan distribusi data sejak awal, proses preprocessing dapat dilakukan secara lebih terarah dan akurat, sehingga model yang dibangun nantinya menjadi lebih andal dan representatif terhadap data sesungguhnya.

#### Informasi Selengkapnya terkait  dataset:
| Jenis | Keterangan |
| ------ | ------ |
| Title | _Bank Marketing Dataset_ |
| Source | [Kaggle](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset) |
| Maintainer | [Janio Martinez Bachmann](https://www.kaggle.com/janiobachmann)|

####  Dataset terdiri dari:
| Jenis | Keterangan |
| ------ | ------ |
|Baris| 11.162 |
| Kolom | 17 |
| Missing Value | 0 |
|Duplikat data| 0 |

Dataset ini tergolong bersih karena tidak memiliki nilai yang hilang maupun duplikat. Seluruh entri dapat langsung digunakan untuk eksplorasi dan pemodelan lebih lanjut.

#### Deteksi Outlier
|Data Outlier| Jumlah |
| -- | -- |
| Age | 171 |
| Balance | 1055 |
| Day | 0 |
| Duration | 636 |
|Campaign | 601 |
| Pdays | 2750 |
| Previous | 1258 |

### Beberapa insight dari outlier:

- Pdays (jumlah hari sejak kontak terakhir): memiliki jumlah outlier tertinggi. Ini mungkin disebabkan oleh nasabah yang sebelumnya belum pernah dihubungi atau baru dihubungi kembali setelah waktu yang sangat lama.
- Balance menunjukkan adanya nasabah dengan saldo ekstrem, baik sangat tinggi maupun negatif.
- Duration juga memiliki penyebaran tinggi, menunjukkan durasi panggilan yang sangat panjang pada beberapa kasus.

### Variabel-variabel pada Bank Marketing Kaggle dataset adalah sebagai berikut:
- age : Usia nasabah (dalam tahun).
- job : Jenis pekerjaan nasabah 
- marital : Status pernikahan nasabah (misalnya: married, single, divorced).
- education : Tingkat pendidikan terakhir nasabah 
- default : Apakah nasabah memiliki kredit macet (default) pada pinjaman sebelumnya (yes/no).
- balance : Saldo rata-rata tahunan dalam rekening nasabah
- housing : Apakah nasabah memiliki pinjaman rumah (yes/no).
- loan : Apakah nasabah memiliki pinjaman pribadi (yes/no).
- contact : Jenis kontak komunikasi terakhir yang digunakan 
- day : Hari terakhir dalam bulan saat kontak dilakukan.
- month : Bulan saat kontak dilakukan.
- duration : Durasi panggilan terakhir dalam detik.
- campaign : Jumlah kontak yang dilakukan selama kampanye marketing saat ini.
- pdays : Jumlah hari sejak nasabah terakhir dihubungi dalam kampanye sebelumnya.
- previous : Jumlah kontak yang dilakukan sebelum kampanye ini.
- poutcome : Hasil dari kampanye marketing sebelumnya (misalnya: success, failure, unknown).
- deposit : Target variabel — apakah nasabah melakukan langganan produk deposito berjangka (yes/no).
 
 
### EDA
- Visualisasi Univariate Analysis
![Univariate Analysis](https://i.ibb.co/LhgZPx01/image.png)
 
```python
plt.figure(figsize=(15, 7))
for i in range(len(num_feature)):
    plt.subplot(2, 4, i + 1)
    sns.histplot(x=num_feature[i], data=df)
    plt.tight_layout()
```
 
- Visualisasi Multivariate Analysis
![Multivariate Analysis](https://i.ibb.co/zTdxtN6r/image.png)
```
sns.pairplot(df[num_feature])
sns.heatmap(df[num_feature].corr(), annot = True)
```
 
## 4. Data Preparation
Tahapan dan Penanganan  data preparation, dilakukan serangkaian proses untuk memastikan data siap digunakan dalam pemodelan machine learning:
 
-   **Remove Outlier**
```
for col in num_feature:
  print(f"Removing outliers from col {col}")
  Q1 = df[col].quantile(0.25)
  Q3 = df[col].quantile(0.75)
  IQR = Q3 - Q1
  fence_high = Q3 + 1.5*IQR
  fence_low = Q1 - 1.5*IQR
  outliers = df[(df[col] < fence_low) | (df[col] > fence_high)]
  df = df[(df[col] >= fence_low) & (df[col] <= fence_high)]
```
 
-  **Feature Engineering**
```
df["balance_group"] = "positive"
df.loc[df["balance"] < 0 , "balance_group"] = "negative"
df.loc[df["balance"] == 0 , "balance_group"] = "empty"
```
-  **One-Hot Encoding**
```
df = pd.get_dummies(df, columns = obj_feature)
```
-  **Train test split**
```
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)
```
 
##### Penjelasan dan alasan
- **Handling Outlier**:  Outlier dihapus menggunakan metode IQR (Interquartile Range) pada fitur numerik seperti age, balance, duration, campaign, pdays, dan previous. Nilai yang berada jauh di luar rentang normal dianggap tidak mewakili populasi umum dan bisa berasal dari kesalahan input atau kasus ekstrem.
- **Feature Engineering**: Fitur baru *balance_group* dibuat berdasarkan nilai balance:
    - Negative -> Saldo negatif
    - Empty -> Saldo nol
    - Positive -> Saldo positif
Tujuannya adalah menyederhanakan informasi numerik ke dalam kategori yang bisa menangkap pola perilaku nasabah yang berbeda.
- **One-Hot Encoding** : Variabel kategorikal seperti job, education, contact, month, dan lainnya diubah ke format numerik menggunakan One-Hot Encoding (pd.get_dummies). Hal ini dilakukan karena sebagian besar algoritma machine learning tidak bisa memproses data kategorikal secara langsung.
- **Train test split** : Sebelum membangun model machine learning, dataset perlu dibagi menjadi dua bagian: data pelatihan (training set) dan data pengujian (testing set). Pembagian ini bertujuan untuk mengevaluasi performa model terhadap data yang belum pernah dilihat sebelumnya.
 
## 5. Modeling
Berdasarkan EDA, data memiliki distribusi yang tidak normal, sehingga digunakan algoritma yang tidak mengasumsikan distribusi normal, seperti Decision Tree dan Random Forest.
 
##### Dalam proyek ini, digunakan dua algoritma Classification untuk memprediksi apakah seorang nasabah akan melakukan subscribe terhadap produk term deposit, yaitu:
1. Decision Tree Classifier: Decision Tree membagi data ke dalam cabang-cabang berdasarkan fitur yang paling memisahkan label target (dalam hal ini: apakah nasabah akan berlangganan term deposit atau tidak). Model ini menggunakan pendekatan rekursif untuk membagi data berdasarkan nilai-nilai fitur yang mengurangi impurity (ketidakteraturan) seperti Gini Impurity atau Entropy. Proses pemisahan dilakukan sampai kondisi tertentu tercapai (misalnya kedalaman maksimal pohon, atau jumlah sampel minimum di daun pohon).Menggunakan default parameters, artinya tidak ada pengaturan manual seperti kedalaman maksimal pohon, jumlah sampel minimal, atau kriteria pemisahan.
Parameter yang Digunakan: Menggunakan default parameters
 
2. Random Forest Classifier (tanpa Hyperparameter Tuning): Random Forest merupakan algoritma ensemble learning berbasis Decision Tree. Model ini membuat banyak decision tree secara acak (random) dengan menggunakan subset dari data dan subset dari fitur, lalu menggabungkan hasil voting dari masing-masing pohon untuk memberikan prediksi akhir. Hal ini meningkatkan akurasi dan mengurangi risiko overfitting yang biasa terjadi pada single Decision Tree.
Parameter yang Digunakan: Menggunakan default parameters
 
3. Random Forest (dengan Hyperparameter Tuning): Sama seperti Random Forest pada umumnya, namun kali ini dilakukan penyesuaian parameter (tuning) untuk mendapatkan kombinasi parameter terbaik. Penyesuaian dilakukan menggunakan teknik GridSearchCV, yang menguji semua kombinasi dari parameter yang ditentukan, dan melakukan validasi silang (cross-validation) sebanyak 5 kali agar hasil evaluasi stabil dan tidak tergantung pada satu subset data.
### Parameter yang Diuji:
```
params = {
    'max_depth': [2,3,5,10,15],
    'min_samples_leaf': [10,20,25,35,50]
}
```
 
- max_depth: Kedalaman maksimum
Fungsi: Mengontrol seberapa dalam pohon boleh tumbuh.
Nilai dicoba: [2, 3, 5, 10, 15] -> model akan diuji dengan kedalaman pohon 2, 3, 5, 10, dan 15.
makin dalam = model bisa belajar lebih banyak (tapi risiko overfitting).
 
- min_samples_leaf: Menentukan jumlah minimum data pada setiap daun pohon, digunakan untuk menghindari pohon tumbuh terlalu spesifik.
 
```
grid_search = GridSearchCV(estimator = RandomForestClassifier(),
                           param_grid = params,
                           cv = 5,
                           scoring = 'roc_auc'
                           )
grid_search.fit(X_train, y_train)
```
- GridSearchCV akan memilih kombinasi terbaik berdasarkan metrik yang digunakan (scoring='roc_auc').
```
rf_best = grid_search.best_estimator_
rf_best.fit(X_train,y_train)
```
 
 
### 1. Decision Tree (DT)
 
##### Kelebihan:
- Sederhana, mudah dipahami dan divisualisasikan.
- Cepat dalam pelatihan dan prediksi.
- Cocok untuk baseline model.
##### Kekurangan:
- Cenderung overfitting pada data training.
- Performa rendah dibanding model ensemble seperti Random Forest.
- Rentan terhadap noise pada data.
 
### 2. Random Forest (RF)
 
##### Kelebihan:
- Performa jauh lebih tinggi dari DT karena merupakan ensemble dari banyak tree.
- Lebih stabil, tidak mudah overfitting.
- Dapat menangani fitur-fitur yang tidak penting dan tetap memberikan hasil prediksi yang baik.
##### Kekurangan:
- Lebih kompleks, lebih lambat dalam training & prediksi dibanding DT.
- Interpretasi lebih sulit dibanding DT.
 
### 3. Random Forest dengan Hyperparameter Tuning (RF-Hyperparameter)
 
##### Kelebihan:
- Sudah dilakukan penyesuaian parameter, sering kali untuk menghindari overfitting.
- Precision tinggi baik untuk mengurangi false positive, artinya lebih selektif dalam memprediksi yang benar-benar positif.
##### Kekurangan:
- Akurasi justru turun dibanding RF default, kemungkinan karena tuning terlalu membatasi model.
- Bisa terjadi underfitting jika parameter terlalu konservatif.
Perlu waktu dan proses lebih dalam pemilihan hyperparameter.
 
### **Model terbaik sebagai Solusi**
**Random Forest Tanpa Hyperparameter Tuning** akan dipilih, dikarenakan model ini yang terbaik dari sisi  presisi yang tertinggi.
 
## 6. Evaluation
Dalam pemasaran berbasis telepon untuk produk deposito bank, precision lebih diprioritaskan dibanding recall karena fokus utama adalah menghindari pemborosan biaya untuk pelanggan yang tidak tertarik, bukan semata-mata menjangkau sebanyak mungkin. Dengan model yang memiliki precision tinggi, bank hanya menghubungi pelanggan yang benar-benar berpotensial berlangganan, sehingga menghemat biaya besar dan tetap menjaga profitabilitas.
 
### Rumus precission:
 
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} \times 100\%$$
 
**Penjelasan**
- TP (True Positive): Jumlah data positif yang diprediksi dengan benar sebagai positif.
- FP (False Positive): Jumlah data negatif yang diprediksi secara tidak benar sebagai positif (Kesalahan Tipe I).
 
**Metrik Evaluasi Precision**
 
| Model | Precision |
| ------ | ------ |
| Decission Tree | 0.72 |
| RandomForest  | 0.80 |
| RandomForest (Tuning) | 0.79 |
 
### Keterkaitan dengan Business Understanding
1. Dengan membangun model klasifikasi seperti Decision Tree dan Random Forest, kita berhasil membuat sistem yang dapat memprediksi apakah seorang nasabah akan berlangganan term deposit atau tidak.
 
2. Melalui evaluasi terhadap berbagai model dengan metrik seperti precision, kita dapat mengidentifikasi model yang memberikan performa terbaik.
 
3. Dengan memilih model yang memiliki precision tinggi, bank dapat lebih fokus pada nasabah yang berpotensi tinggi untuk berlangganan. Hal ini meningkatkan efektivitas pemasaran dan mengurangi jumlah panggilan telemarketing yang cenderung tidak efisiena.
 
### Pencapaian Goals:
 
1. Model berhasil mengklasifikasikan nasabah ke dalam dua kategori: berpotensi berlangganan atau tidak.
 
2. Proses perbandingan antar model dan penerapan hyperparameter tuning memberikan insight tentang model mana yang paling optimal berdasarkan metrik precision.
 
3. Jumlah nasabah yang perlu dihubungi dapat dikurangi secara signifikan, karena hanya target dengan potensi tinggi yang diprioritaskan. Ini berdampak pada efisiensi biaya pemasaran dan peningkatan ROI dalam kampanye pemasaran term deposit.
 
### Dampak dari Solution Statements:
 
1. Implementasi model klasifikasi telah menunjukkan hasil dalam mengotomatisasi dan meningkatkan proses seleksi nasabah yang memiliki potensial.
 
2. Evaluasi menggunakan precision memastikan bahwa dari seluruh nasabah yang diprediksi akan berlangganan, sebagian besar benar-benar tertarik. ini merupakan salah satu hal penting untuk menghindari pemborosan sumber daya dan waktu.
 
3. Model terbaik yang terpilih berdasarkan evaluasi tersebut secara langsung mendukung strategi pemasaran yang lebih terfokus, terarah, dan hemat biaya.
 
 
 
**---Ini adalah bagian akhir laporan---**
 
_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.