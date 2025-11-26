# YOLOv8 Face Recognition (Deteksi + Pengenalan Wajah)

Proyek ini menyediakan implementasi lengkap untuk melakukan deteksi wajah dan pengenalan wajah menggunakan YOLOv8. Sistem bekerja dalam dua tahap utama:

1. YOLOv8-Det (yolov8n.pt) untuk mendeteksi lokasi wajah.
2. YOLOv8-Cls (yolov8n-cls.pt) untuk mengenali identitas (face recognition) dari wajah yang terdeteksi.

Metode ini menggunakan YOLOv8 secara penuh, tanpa FaceNet, tanpa embedding, dan tanpa KNN.

---

## 1. Fitur Utama
- Deteksi wajah realtime menggunakan YOLOv8.
- Pengenalan wajah menggunakan YOLOv8 Classification.
- Training sederhana hanya dengan menyusun dataset folder-per-orang.
- Pipeline dua tahap: deteksi → pengenalan.
- Bisa dijalankan di laptop biasa (CPU) maupun GPU.
- Struktur project bersih dan mudah dikembangkan.

---

## 2. Struktur Folder Project
------------------------------------------------------------
Struktur Folder
------------------------------------------------------------

project/
 ├── datasets/
 │     ├── Asep/
 │     ├── Budi/
 │     ├── Siti/
 │     └── ...
 ├── train_recognition.py
 ├── recognize.py
 ├── detect_and_recognize.py
 ├── requirements.txt
 └── README.md
------------------------------------------------------------

Penjelasan:
- datasets/ → folder dataset berisi foto wajah per orang.
- train_recognition.py → script training YOLOv8 Classification.
- recognize.py → pengenalan wajah dari file gambar.
- detect_and_recognize.py → deteksi + pengenalan wajah realtime via webcam.
- requirements.txt → daftar package utama.
- README.md → dokumentasi project.

---

## 3. Persiapan Dataset

Dataset harus disusun dalam format folder per identitas seperti:

datasets/
  /Asep/
      img1.jpg
      img2.jpg
  /Budi/
      foto1.png
      foto2.jpeg
  /Siti/
      foto01.jpg
      foto02.png

Aturan dataset:
- Nama folder akan menjadi label identitas.
- Setiap folder berisi foto wajah orang tersebut.
- Minimal 5–10 foto per orang untuk hasil optimal.
- Foto lebih variatif = akurasi lebih baik.

Tidak perlu anotasi bounding box karena YOLOv8 Classification mengerti otomatis struktur folder tersebut.

---

## 4. Instalasi Dependensi

Jalankan perintah:

pip install -r requirements.txt

Isi file requirements.txt:

ultralytics
opencv-python
numpy

Keterangan package:
- ultralytics → YOLOv8 (deteksi & klasifikasi)
- opencv-python → pembacaan kamera & pengolahan gambar
- numpy → utilitas array numerik

---

## 5. Training YOLOv8 Classification (Face Recognition)

Jalankan:

python train_recognition.py

Setelah training selesai, model akan disimpan otomatis di:

runs/classify/train/weights/best.pt

Model ini digunakan untuk pengenalan wajah pada tahap inference.

---

## 6. Pengenalan Wajah dari Gambar

Jalankan:

python recognize.py

Fungsi script:
- Membaca satu file gambar
- Memprediksi identitas wajah
- Menampilkan nama dan confidence

---

## 7. Deteksi + Pengenalan Wajah Realtime (Webcam)

Jalankan:

python main.py

Fungsi script:
1. YOLOv8n mendeteksi wajah dari kamera.
2. Crop wajah dikirim ke model YOLOv8n-cls.
3. Model memberikan nama + confidence.
4. Ditampilkan secara realtime.

Tekan Q untuk keluar.

---

## 8. Penjelasan File Utama

### train_recognition.py
Script training YOLOv8 Classification menggunakan folder dataset.

### detect_and_recognize.py
Pipeline dua model:
- YOLOv8 deteksi wajah
- YOLOv8 klasifikasi identitas

### recognize.py
Prediction identitas wajah dari satu gambar.

---

## 9. Tips dan Rekomendasi
- Semakin banyak foto per orang → performa lebih baik.
- Pencahayaan sangat mempengaruhi hasil.
- Jika ingin performa lebih tinggi:
  - gunakan yolov8s-cls.pt atau yolov8m-cls.pt
- Gunakan GPU untuk percepatan training.

---

## 10. Roadmap
- [ ] Integrasi FastAPI untuk layanan API face recognition
- [ ] Penambahan sistem absensi otomatis
- [ ] Logging database untuk identifikasi harian
- [ ] Tracking menggunakan DeepSORT
- [ ] Augmentasi otomatis dataset

---

## 11. Commit Message Rekomendasi

feat: add full YOLOv8 face recognition system (detection + classification)

---

## 12. Lisensi
Proyek ini bebas digunakan untuk keperluan pembelajaran, penelitian, dan pengembangan non-komersial.

