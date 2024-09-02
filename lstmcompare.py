import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'train_losses' not in st.session_state:
    st.session_state['train_losses'] = []
if 'mape_losses' not in st.session_state:
    st.session_state['mape_losses'] = []
if 'rmse_losses' not in st.session_state:
    st.session_state['rmse_losses'] = []

# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), self.hidden_layer_size).requires_grad_()
        c_0 = torch.zeros(1, x.size(0), self.hidden_layer_size).requires_grad_()
        lstm_out, _ = self.lstm(x, (h_0.detach(), c_0.detach()))
        lstm_out = self.dropout(lstm_out)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

# GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, dropout):
        super(GRUModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.gru = nn.GRU(input_size, hidden_layer_size, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), self.hidden_layer_size).requires_grad_()
        gru_out, _ = self.gru(x, h_0.detach())
        gru_out = self.dropout(gru_out)
        predictions = self.fc(gru_out[:, -1, :])
        return predictions

# Fungsi untuk menghitung MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    mask = y_true != 0  # Hindari pembagian dengan 0
    return torch.mean(torch.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Fungsi untuk menghitung RMSE
def root_mean_square_error(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

# Fungsi untuk menghitung RMSE pada skala asli
def root_mean_square_error_original(y_true, y_pred, scaler):
    y_true_original = scaler.inverse_transform(y_true.reshape(-1, 1))
    y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1))
    return np.sqrt(np.mean((y_true_original - y_pred_original) ** 2))

# Load dataset
def load_data(file, stock_symbol=None, start_date=None, end_date=None):
    if stock_symbol:
        df = yf.download(stock_symbol, start=start_date, end=end_date)
    elif file:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, parse_dates=True, index_col='Date')
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, parse_dates=True, index_col='Date')
        else:
            st.error("File tidak didukung. Unggah file CSV atau Excel.")
            return None
    else:
        return None
    return df

# Prepare time series data
def prepare_time_series_data(df, feature, look_back):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[feature].values.reshape(-1, 1))
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return torch.Tensor(X), torch.Tensor(y), scaler

# Train model
def train_model_torch(X_train, y_train, method, hidden_layer_size, epochs, batch_size, dropout, learning_rate):
    input_size = 1
    output_size = 1

    if method == 'LSTM':
        model = LSTMModel(input_size, hidden_layer_size, output_size, dropout=dropout)
    else:
        model = GRUModel(input_size, hidden_layer_size, output_size, dropout=dropout)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    mape_losses = []
    rmse_losses = []

    model.train()
    for epoch in range(epochs):
        epoch_losses = []
        epoch_mape = []
        epoch_rmse = []
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.view(-1, 1))
            mape = mean_absolute_percentage_error(y_batch, y_pred)
            rmse = root_mean_square_error(y_batch, y_pred)

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_mape.append(mape.item())
            epoch_rmse.append(rmse.item())

        train_losses.append(np.mean(epoch_losses))
        mape_losses.append(np.mean(epoch_mape))
        rmse_losses.append(np.mean(epoch_rmse))

    return model, train_losses, mape_losses, rmse_losses

# Fungsi untuk prediksi beberapa hari ke depan
def predict_future(model, last_data, days_ahead, scaler):
    model.eval()
    predictions = []
    current_data = last_data
    for _ in range(days_ahead):
        with torch.no_grad():
            prediction = model(current_data.view(1, -1, 1))
            predictions.append(prediction.item())
            current_data = torch.cat((current_data[1:], prediction), dim=0)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

# Fungsi utama Streamlit
def main():
    st.title("Time Series Forecasting menggunakan LSTM/GRU")

    data_source = st.radio("Pilih sumber data:", ("Upload file", "Yahoo Finance"))

    if data_source == "Upload file":
        file = st.file_uploader("Unggah file dataset (CSV atau Excel)", type=['csv', 'xlsx'])
        st.markdown("<p style='color:blue;'>Catatan: Pastikan data dalam bentuk integer atau float dan tidak ada missing value!</p>", unsafe_allow_html=True)
        if file is not None:
            df = load_data(file)
            if df is not None:
                st.session_state['df'] = df
    elif data_source == "Yahoo Finance":
        stock_symbol = st.text_input("Masukkan simbol saham (misalnya AAPL):")
        st.markdown("<p style='color:blue;'>Catatan: Pastikan ticker atau simbol data benar!</p>", unsafe_allow_html=True)
        start_date = st.date_input("Pilih tanggal mulai:", value=pd.to_datetime('2020-01-01'))
        end_date = st.date_input("Pilih tanggal akhir:", value=pd.to_datetime('today'))
        if st.button("Ambil Data"):
            df = load_data(None, stock_symbol, start_date, end_date)
            if df is not None:
                st.session_state['df'] = df

    df = st.session_state.get('df', None)
    if df is not None:
        st.subheader("Data yang Diupload atau Diambil")
        st.write(df)

        # Pilih fitur dan target
        st.subheader("Pilih Fitur Time Series")
        feature = st.selectbox("Pilih fitur yang akan digunakan untuk peramalan:", options=df.columns)

        if feature:
            # Tanggal terakhir dari dataset
            last_date = df.index[-1]

            # Pilih model
            method = st.radio("Pilih metode:", ("LSTM", "GRU"))

            # Pilih parameter model
            look_back = st.slider("Berapa banyak langkah ke belakang untuk pelatihan model?", 1, 60, 30)
            hidden_layer_size = st.slider("Ukuran Hidden Layer:", 10, 100, 64)
            epochs = st.slider("Jumlah epoch:", 10, 500, 50)
            batch_size = st.slider("Ukuran batch:", 8, 128, 32)
            learning_rate = st.slider("Learning Rate:", 0.001, 0.01, 0.001, step=0.0001)
            dropout = st.slider("Dropout:", 0.0, 0.5, 0.2)
            days_ahead = st.slider("Berapa hari ke depan yang ingin diramalkan?", 1, 30, 7)

            # Latih model dan lakukan peramalan
            if st.button("Latih Model dan Ramalkan"):
                X_train, y_train, scaler = prepare_time_series_data(df, feature, look_back)
                model, train_losses, mape_losses, rmse_losses = train_model_torch(
                    X_train, y_train, method, hidden_layer_size, epochs, batch_size, dropout, learning_rate)

                st.session_state['model'] = model

                # Plotting grafik loss, MAPE, dan RMSE dalam satu grafik
                st.subheader("Grafik Loss, MAPE, dan RMSE selama pelatihan")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(epochs)), y=train_losses, mode='lines', name='Train Loss'))
                fig.add_trace(go.Scatter(x=list(range(epochs)), y=mape_losses, mode='lines', name='MAPE'))
                fig.add_trace(go.Scatter(x=list(range(epochs)), y=rmse_losses, mode='lines', name='RMSE'))
                st.plotly_chart(fig)

                # Plot hasil data latih vs data aktual
                st.subheader("Hasil Prediksi pada Data Latih")
                with torch.no_grad():
                    y_pred_train = model(X_train).numpy()
                y_pred_train = scaler.inverse_transform(y_pred_train)
                y_train_actual = scaler.inverse_transform(y_train.view(-1, 1))

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index[-len(y_train):], y=y_train_actual.flatten(), mode='lines', name='Data Aktual'))
                fig.add_trace(go.Scatter(x=df.index[-len(y_train):], y=y_pred_train.flatten(), mode='lines', name='Data Latih', line=dict(color='red')))
                st.plotly_chart(fig)

                # Menghitung RMSE pada skala asli
                rmse_original = root_mean_square_error_original(y_train.numpy(), y_pred_train, scaler)
                st.subheader(f"RMSE pada skala asli: {rmse_original:.4f}")

                # Peramalan beberapa hari ke depan
                st.subheader(f"Prediksi {days_ahead} hari ke depan")
                last_data = X_train[-1].clone().detach()  # Mengambil data terakhir dari data latih
                predictions = predict_future(model, last_data, days_ahead, scaler)
                future_dates = pd.date_range(last_date + pd.DateOffset(1), periods=days_ahead)


                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df[feature], mode='lines', name='Data Historis'))
                fig.add_trace(go.Scatter(x=future_dates, y=predictions.flatten(), mode='lines', line=dict(color='red'), name='Prediksi Masa Depan'))
                st.plotly_chart(fig)

                # Menampilkan hasil prediksi dalam bentuk tabel
                st.subheader("Tabel Hasil Peramalan")
                pred_df = pd.DataFrame({
                    'Tanggal': future_dates,
                    'Hasil Peralaman': predictions.flatten()
                })
                st.write(pred_df)

if __name__ == "__main__":
    main()
