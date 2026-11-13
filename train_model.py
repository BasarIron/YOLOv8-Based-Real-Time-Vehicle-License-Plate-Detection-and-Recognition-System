import torch
from ultralytics import YOLO
import time

def main():
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"-> Kullanılan Donanım: {'NVIDIA GPU (CUDA)' if device==0 else 'CPU'}")

    model = YOLO("yolov8n.pt")

    print("-> Model eğitimi başlatılıyor...")
    start_time = time.time()
    
    results = model.train(
        data="dataset/data.yaml",   # Veri seti konfigürasyonu
        epochs=30,                  # Eğitim tur sayısı 
        batch=16,                   # Batch boyutu 
        imgsz=640,                  # Girdi görüntü boyutu
        optimizer="AdamW",          # Ödev için seçilen Optimizer
        lr0=0.001,                  # Başlangıç öğrenme oranı 
        device=device,              # Ekran kartı kullanımı
        project="plaka_output",     # Çıktıların kaydedileceği klasör
        name="yolov8_run"
    )
    
    train_duration = (time.time() - start_time) / 60
    print(f"-> Eğitim tamamlandı! Süre: {train_duration:.2f} dakika.")

    print("-> Test kümesi üzerinde başarım metrikleri hesaplanıyor...")
    metrics = model.val(data="dataset/data.yaml", split="test", device=device)
    
    print("\n================ METRİK RAPORU ================")
    print(f"mAP@50 (IoU=0.50): {metrics.box.map50:.4f}")
    print(f"mAP@50-95        : {metrics.box.map:.4f}")
    print(f"Precision        : {metrics.box.mp:.4f}")
    print(f"Recall           : {metrics.box.mr:.4f}")
    print(f"Ortalama Tahmin Süresi: {metrics.speed['inference']:.2f} ms")
    print("================================================")
    print("En iyi model ağırlığı şu konumda saklandı: plaka_output/yolov8_run/weights/best.pt")

if __name__ == '__main__':
    main()