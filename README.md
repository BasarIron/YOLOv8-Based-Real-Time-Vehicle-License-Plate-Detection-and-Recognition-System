# 🚗 YOLOv8 Tabanlı Gerçek Zamanlı Araç Plaka Tespit ve Tanıma Sistemi

Bu proje, **Bilgisayarlı Görü (Computer Vision) – Derin Öğrenme Uygulaması** ödevi kapsamında geliştirilmiştir. Açık kaynaklı bir veri seti kullanılarak dış ortam araç görüntülerinden gerçek zamanlı plaka tespiti yapan özelleştirilmiş bir **YOLOv8** modeli eğitilmiş ve **Streamlit** tabanlı interaktif bir web arayüzü ile canlı uygulamaya dönüştürülmüştür.

---

## 📋 1. Proje Raporu Özeti (Şablon Tablosu)

Ödev yönergesinde belirtilen standart proje raporu şablonu ve projemize ait teknik parametreler aşağıda özetlenmiştir:

| Alan | Açıklama / Proje Değerleri |
| :--- | :--- |
| **Project Title** | YOLOv8 Tabanlı Gerçek Zamanlı Araç Plaka Tespit ve Tanıma Sistemi |
| **Problem / Task** | Nesne Tespiti (Object Detection) ve Optik Karakter Tanıma (OCR) |
| **Dataset** | Roboflow - plaka Computer Vision Model (~19.153 Görüntü) |
| **Dataset URL** | [Roboflow Dataset Linki](https://universe.roboflow.com/plaka-nznwq/plaka-ta8g7/dataset/2/download) |
| **Model Architecture** | YOLOv8n (CSPDarknet53 Omurga + PANet Boyun + Anchor-Free Tespit Başlığı) |
| **Input Size** | `640 × 640 × 3` (RGB) |
| **Hyperparameters** | `Epochs: 30`, `Batch Size: 16`, `Initial Learning Rate (lr0): 0.001`, `Weight Decay: 0.0005` |
| **Optimizer** | `AdamW` (Ağırlık çürümesini gradyan güncellemesinden ayırarak aşırı öğrenmeyi / overfitting'i engellediği ve CNN/YOLO mimarilerinde SGD'ye göre daha hızlı yakınsadığı için seçilmiştir) |
| **Activation / Output Function** | Gizli katmanlarda **SiLU (Swish)** aktivasyon fonksiyonu; çıkış katmanında sınıf olasılıkları için **Sigmoid** aktivasyon fonksiyonu kullanılmıştır. |
| **Loss Function** | CIoU Loss + DFL (Distribution Focal Loss) (Kutu koordinat regresyonu için) ve Binary Cross-Entropy (BCE) Loss (Sınıflandırma kaybı için) bileşimi kullanılmıştır. |
| **Evaluation Metrics** | IoU, mAP@50, mAP@50-95, Precision, Recall |
| **Training Environment** | Lokal GPU (NVIDIA GeForce RTX 2070 Super, PyTorch 2.6.0+cu124, Python 3.13) |
| **Best Performance** | `mAP@50: 0.8706`, `mAP@50-95: 0.5783`, `Precision: 0.9333`, `Recall: 0.8800` |
| **Inference Time** | `5.28 ms / görüntü` (NVIDIA RTX 2070 Super GPU) |
| **Deployment Technology** | Streamlit |
| **Repository** | [GitHub Repository](https://github.com/BasarIron/YOLOv8-Based-Real-Time-Vehicle-License-Plate-Detection-and-Recognition-System.git) |

---

## 🎯 2. Ödevin Amacı ve Görev Tanımı (Task)
Bu çalışmanın temel amacı, farklı ışık, açı ve mesafe koşullarındaki araç fotoğrafları içerisinden plaka bölgesini yüksek doğrulukla tespit eden uçtan uca bir derin öğrenme modeli geliştirmektir. Bilgisayarlı görü görevlerinden **Nesne Tespiti (Object Detection)** seçilmiş ve model çıktıları geometrik en-boy oranı (aspect ratio) filtreleriyle desteklenerek hatalı pozitif (false positive) tahminler minimize edilmiştir.

---

## 🧠 3. Model Mimarisi ve Eğitim Ayarları
Problemin gerçek zamanlı çalışabilmesi ve yüksek doğruluk sunması adına tek aşamalı (single-stage) nesne tespit mimarisi olan **YOLOv8n** tercih edilmiştir.

- **Hiperparametre Seçimi:** Model `640x640` çözünürlükte, `16` batch size, `30` epoch ve `0.001` başlangıç öğrenme oranı (`lr0`) ile eğitilmiştir. 
- **Optimizer:** Aşırı öğrenmeyi (overfitting) engellemek ve hızlı yakınsama sağlamak amacıyla **AdamW** optimizasyon algoritması (`weight_decay=0.0005`) kullanılmıştır.
- **Aktivasyon ve Kayıp (Loss) Fonksiyonları:** Ağın omurgasında **SiLU** aktivasyonu kullanılmıştır. Kayıp fonksiyonu olarak ise kutu örtüşmesini optimize eden **CIoU Loss**, kutu sınırlarını hassaslaştıran **DFL (Distribution Focal Loss)** ve varlık tespiti için **Binary Cross-Entropy (BCE)** birlikte kullanılmıştır.

---

## 📂 4. Veri Seti Yapısı (Dataset)
Projede yaklaşık **19.153 görüntü** içeren Roboflow açık kaynaklı plaka veri seti kullanılmıştır. Veri seti modelin genelleme yeteneğini ölçmek için standart alt kümelerine ayrılmıştır:
- **`train/`**: Modelin ağırlıklarının güncellendiği eğitim kümesi.
- **`valid/`**: Eğitim esnasında her epoch sonunda başarımını doğrulayan küme.
- **`test/`**: Modelin eğitimde hiç görmediği, nihai başarımın (`mAP@50: 0.8706`) ölçüldüğü test kümesi.

---

## 📊 5. Performans Değerlendirmesi
Lokal GPU ortamında (NVIDIA GeForce RTX 2070 Super) elde edilen deneysel sonuçlar şunlardır:
- **Başarım Metrikleri:** `mAP@50: 0.8706`, `mAP@50-95: 0.5783`, `Precision: 0.9333`, `Recall: 0.8800`.
- **Tahmin (Inference) Süresi:** Donanım hızlandırma sayesinde ortalama **5.28 ms / görüntü** gibi üstün bir gerçek zamanlı performans elde edilmiştir.

---

## 📈 5.1. Eğitim Süreci ve Başarım Grafikleri Analizi

Modelin 30 epoch boyunca gösterdiği öğrenme dinamikleri, doğruluk artışları ve genelleme yeteneği eğitim loglarından elde edilen grafiklerle doğrulanmıştır.

### A. mAP@50 ve mAP@50-95 Gelişimi
![mAP Metrikleri](mAPmetrics.png)

* **mAP@50 (Mor Eğri):** İlk epoch'lardaki başlangıç dalgalanmalarının ardından model hızla öğrenmiş, eğitim ilerledikçe istikrarlı bir yükseliş trendi yakalayarak **0.975** seviyelerine ulaşmıştır (Füzyon/Final test aşamasında ~0.974).
* **mAP@50-95 (Yeşil Eğri):** Farklı IoU eşiklerinde modelin katı doğruluk payını gösteren bu metrik, 30 epoch boyunca düzenli bir artış göstererek **0.606** değerine tırmanmıştır. Bu durum, modelin kutu konumlandırmasında (localization) yüksek hassasiyet kazandığını kanıtlamaktadır.

### B. Precision (Kesinlik) ve Recall (Duyarlılık) Eğrileri
![Precision ve Recall](precission_recall.jpg)

* **Precision - Kesinlik (Mavi Eğri):** Modelin yaptığı tahminlerin ne kadarının doğru olduğunu gösterir. Eğitim sürecinin sonunda **0.963** gibi çok yüksek bir orana uluşarak yanlış pozitif (false positive) oranını minimumda tuttuğunu kanıtlamıştır.
* **Recall - Duyarlılık (Turuncu Eğri):** Gerçek plaka nesnelerinin yüzde kaçının model tarafından başarıyla yakalandığını ifade eder. Eğitimin sonunda **0.958** seviyesine sabitlerek modelin sahadaki nesneleri kaçırma oranının son derece düşük olduğunu göstermiştir.

---

## 📸 6. Gerçek Görüntülerle Test ve Değerlendirme (5 Fotoğraf Analizi)
Ödev kapsamında telefon kamerası ile dış ortamda çekilen 5 farklı araç fotoğrafı sisteme girdi olarak verilmiş ve modelin başarımı test edilmiştir:

1. **Test Görüntüsü 1 (Ön Cephe - Net Açılı):** Plaka bölgesi yüksek güven skoruyla tam sınırlarından tespit edilmiştir.
2. **Test Görüntüsü 2 (Yan/Çapraz Açı):** Perspektif bozulmasına rağmen model plakayı başarıyla lokalize etmiştir.
3. **Test Görüntüsü 3 (Parlak Gün Işığı / Yansıma):** Kaput ve tampon üzerindeki güneş yansımalarına rağmen plaka çerçevesi doğru ayrılmıştır.
4. **Test Görüntüsü 4 (Uzak Mesafe Çekimi):** Piksel alanı küçülmesine rağmen tespit başarılı bir şekilde gerçekleştirilmiştir.
5. **Test Görüntüsü 5 (Zorlu Işık / Gölgeli Ortam):** Karanlık ortamlarda dahi model kararlılığını korumuştur.

---

