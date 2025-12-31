import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import random
import os
import seaborn as sns

NASA_SMAP = 'A-1.npy' 
Model_file = 'uzay_copu_transformer_modeli.pth'
impact_file = 'impact.csv'
test_samples = 500  
window = 200
Noise = 0.23  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
class AnomalyTransformer(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=2):
        super(AnomalyTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 200, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(d_model * 200, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer_encoder(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.sigmoid(x)

# veri yükleme
def load_data():
    try:
        if NASA_SMAP.endswith('.npy'):
            real_data = np.load(NASA_SMAP)
            if len(real_data.shape) > 1: real_data = real_data[:, 0]
        elif NASA_SMAP.endswith('.csv'):
            df = pd.read_csv(NASA_SMAP)
            real_data = df.iloc[:, 0].values
        else:
            print("Desteklenmeyen dosya formatı.")
            return None, None, None
        
        real_data = (real_data - np.mean(real_data)) / (np.std(real_data) + 1e-6)
    except Exception as e:
        print(f"Gerçek veri yüklenemedi: {e}")
        return None, None, None

    try:
        df_imp = pd.read_csv(impact_file)
        if 'Gyro_X' in df_imp.columns: raw = df_imp['Gyro_X'].values
        else: raw = df_imp.iloc[:, 1].values
        
        start = np.argmax(np.abs(raw) > 0.1)
        trim = raw[start:].copy()
        if len(trim) > 200: trim = trim[:200]

        t = np.arange(len(trim))
        damping_curve = np.exp(-0.05 * t) * np.cos(0.2 * t)
        temp = trim * damping_curve
        impact_template = temp / (np.max(np.abs(temp)) + 1e-9)
        
    except Exception as e:
        print(f"Çarpma verisi bulunamadı: {e}")
        return None, None, None
        
    return real_data, impact_template, trim

# test seti oluşturma
def create_test_set(real_data, impact_template, num_samples):
    X_test = []
    y_test = []
    
    for _ in range(num_samples):
        idx = random.randint(0, len(real_data) - window - 1)
        sample = real_data[idx : idx + window].copy()
        
        label = 0 
        if random.random() > 0.5:
            label = 1
            scale = random.uniform(1, 1.5) 
            L = len(impact_template)
            if L < window:
                insert_idx = random.randint(10, window - L - 10)
                sample[insert_idx : insert_idx + L] += impact_template * scale
            else:
                sample[10:150] += impact_template[:140] * scale
        
        # noise enjection
        noise = np.random.normal(0, Noise, len(sample))
        sample += noise
                
        X_test.append(sample)
        y_test.append(label)
        
    return np.array(X_test), np.array(y_test)
def plot_physical_proof(real_data, raw_impact, processed_impact):
    pass 

# main loop
def main():
    real_data, impact_template, raw_impact = load_data()
    if real_data is None: return

    # Veri setini oluşturma
    X_test, y_test = create_test_set(real_data, impact_template, test_samples)
    X_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(device)
    
    if not os.path.exists(Model_file):
        print(f"HATA: '{Model_file}' bulunamadı.")
        return
        
    model = AnomalyTransformer().to(device)
    try:
        model.load_state_dict(torch.load(Model_file, map_location=device))
    except:
        print("Model yüklenemedi.")
        return

    model.eval()
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy().flatten()
    
    y_pred = (preds > 0.85).astype(int)
    
    # metriklerin hesaplanması
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # rapor
    print("\n" + "="*40)
    print(f"MODEL PERFORMANS RAPORU")
    print("="*40)
    print(f"Test Örnek Sayısı : {test_samples}")
    print(f"Gürültü Seviyesi  : {Noise}")
    print("-" * 40)
    print(f"Accuracy : %{acc*100:.2f}")
    print(f"Precision: %{prec*100:.2f}")
    print(f"Recall   : %{rec*100:.2f}")
    print(f"F1-Score : %{f1*100:.2f}")
    print("="*40 + "\n")
    
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Çarpma (Impact)'],
                yticklabels=['Normal', 'Çarpma (Impact)'])
    plt.title('Confusion Matrix', fontsize=14)
    plt.ylabel('Gerçek Durum', fontsize=12)
    plt.xlabel('Model Tahmini', fontsize=12)
    plt.show()

if __name__ == "__main__":
    main()
# false positive analysis
def analyze_false_positives(model, X_tensor, y_test, X_test):
    # Tahminleri alma
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy().flatten()
    y_pred = (preds > 0.85).astype(int)
    
    # False Positive İndekslerini Bulma
    fp_indices = [i for i in range(len(y_test)) if y_test[i] == 0 and y_pred[i] == 1]
    
    print(f"\nToplam Yanlış Alarm (False Positive) Sayısı: {len(fp_indices)}")
    
    if len(fp_indices) > 0:
        plt.figure(figsize=(10, 4))
        # İlk hatalı örneği çizdirme
        idx = fp_indices[0]
        plt.plot(X_test[idx], 'k-', label='Sinyal (Sadece Gürültü)')
        plt.title(f"Hatalı Tespit Örneği (False Positive) - İndeks: {idx}", fontweight='bold')
        plt.xlabel("Zaman")
        plt.ylabel("Genlik")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        print("Oluşan grafik, modelin 'çarpma' sandığı ama aslında sadece yüksek gürültü olan sinyaldir.")
    else:
        print("yanlış alarm yok.")
# main loop
if __name__ == "__main__":
    main()
    
    real_data, impact_template, _ = load_data()
    if real_data is not None:
        X_test, y_test = create_test_set(real_data, impact_template, test_samples)
        X_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(device)
        
        model = AnomalyTransformer().to(device)
        if os.path.exists(Model_file):
            model.load_state_dict(torch.load(Model_file, map_location=device))
            model.eval()
            analyze_false_positives(model, X_tensor, y_test, X_test)
