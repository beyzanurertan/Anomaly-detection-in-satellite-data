# Anomaly-detection-in-satellite-data
impact.csv- Simulink ortamında modellenen çarpma anı verisidir.
nominal_data.csv- Matlab ortamında animasyon ve görselleştirme ile gerçek veri üzerinde çalışmadan önce modelin denendiği nominal uydu verisini içerir.
nominal_satellite.m dosyası bu veriyi oluşturmakta kullanılmıştır.
A-1.npy- SMAP uydusuna ait gerçek veri dosyasıdır. Bu veri ve impact.csv verisi python ortamında birleştirilerek eğitim verisi oluşturulmuştur.
uydu_copu_tranformer_modeli.pth- model dosyasıdır.
nasa_confusion_matrix.py- Bu .py dosyası, modelin başarım sonuçlarının elde edildiği kodları içerir.
