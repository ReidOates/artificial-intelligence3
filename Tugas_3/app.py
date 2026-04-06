import os
import io
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Render matplotlib without a display
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)

# Konfigurasi Upload
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__name__)), 'static', 'uploads')
DEFAULT_DATASET = os.path.join(os.path.dirname(os.path.abspath(__name__)), 'dataset', 'aapl.us.txt')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(DEFAULT_DATASET), exist_ok=True)

def process_data_and_predict(file_path, input_open=None):
    try:
        # Membaca dataset
        df = pd.read_csv(file_path)
        
        if 'Open' not in df.columns or 'Close' not in df.columns:
            return {"error": "Format dataset salah. Harus memiliki kolom 'Open' dan 'Close'."}
            
        # Hanya gunakan observasi tanpa nilai kosong
        df = df.dropna(subset=['Open', 'Close'])
        
        # Variabel Independen & Dependen
        X = df[['Open']]
        Y = df['Close']
        
        # Membagi Data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        # Membangun dan Melatih Model
        model = LinearRegression()
        model.fit(X_train, Y_train)
        
        # Evaluasi
        Y_pred = model.predict(X_test)
        mae = mean_absolute_error(Y_test, Y_pred)
        mse = mean_squared_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)
        
        # Prediksi Custom (Jika pengguna menginput manual)
        custom_pred = None
        if input_open:
            try:
                input_val = float(input_open)
                custom_pred = model.predict(pd.DataFrame({'Open': [input_val]}))[0]
            except ValueError:
                pass
                
        # Visualisasi Menggunakan Matplotlib/Seaborn
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        
        # Plot Data Training
        plt.scatter(X_test, Y_test, color='#3498db', alpha=0.5, label="Data Aktual (Test)")
        # Plot Garis Regresi
        plt.plot(X_test, Y_pred, color='#e74c3c', linewidth=2, label="Regresi Linear (Prediksi)")
        
        if custom_pred:
            plt.scatter([input_val], [custom_pred], color='#f1c40f', s=150, zorder=5, edgecolor='black', label=f"Prediksi X={input_val}")
            
        plt.title('Regresi Linear: Harga Pembukaan (Open) vs Harga Penutupan (Close)', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Harga Buka / Open Price ($)', fontsize=12)
        plt.ylabel('Harga Tutup / Close Price ($)', fontsize=12)
        plt.legend(loc='upper left')
        
        # Convert plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        # Buat data sampel untuk tabel jurnal (15 baris pertama dari data tes)
        table_data = []
        for i in range(min(15, len(X_test))):
            original_idx = X_test.index[i]
            date_val = df.loc[original_idx, 'Date'] if 'Date' in df.columns else str(i+1)
            table_data.append({
                'date': date_val,
                'open': round(float(X_test.iloc[i].values[0]), 2),
                'actual_close': round(float(Y_test.iloc[i]), 2),
                'pred_close': round(float(Y_pred[i]), 2)
            })
            
        # Evaluasi pada data latih (untuk cek overfitting/underfitting)
        train_pred = model.predict(X_train)
        r2_train = r2_score(Y_train, train_pred)
            
        return {
            "success": True,
            "mae": round(mae, 2),
            "mse": round(mse, 2),
            "r2": round(r2, 4),
            "r2_train": round(r2_train, 4),
            "mae_exact": "{:.7e}".format(mae),
            "mse_exact": "{:.7e}".format(mse),
            "r2_exact": "{:.7e}".format(r2),
            "r2_train_exact": "{:.7e}".format(r2_train),
            "plot_url": plot_url,
            "input_open": input_open,
            "custom_pred": round(custom_pred, 2) if custom_pred else None,
            "total_data": len(df),
            "table_data": table_data
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_open = request.form.get('input_open')
    
    file_path = DEFAULT_DATASET
    filename = "Apple Stock (Default)"
    
    # Cek jika pengguna upload file sendiri
    if 'file' in request.files:
        file = request.files['file']
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
    # Pastikan default dataset ada jika pengguna tidak upload
    if file_path == DEFAULT_DATASET and not os.path.exists(DEFAULT_DATASET):
        return render_template('index.html', error="Dataset default tidak ditemukan! Harap upload dataset.")

    result = process_data_and_predict(file_path, input_open)
    
    if "error" in result:
        return render_template('index.html', error=result['error'])
        
    return render_template('result.html', result=result, filename=filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
