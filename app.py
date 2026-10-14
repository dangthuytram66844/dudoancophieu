import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Dự đoán LSTM", layout="wide")
st.title("📈 Dự đoán & Biểu đồ Tăng Giảm Cổ phiếu (LSTM)")

# 1. KHU VỰC NHẬP THÔNG SỐ (Mã, Ngày bắt đầu, Ngày kết thúc)
st.sidebar.header("Thông số cấu hình")
ticker = st.sidebar.text_input("Mã cổ phiếu (Ví dụ: AAPL, MSFT, VNM.VN):", "AAPL").upper()
start_date = st.sidebar.date_input("Ngày bắt đầu:", datetime(2018, 1, 1))
end_date = st.sidebar.date_input("Ngày kết thúc:", datetime.today())

# Tải dữ liệu từ Yahoo Finance
data = yf.download(ticker, start=start_date, end=end_date)

if data.empty:
    st.error("Không tìm thấy dữ liệu. Vui lòng kiểm tra lại mã hoặc ngày tháng!")
else:
    # Lấy giá đóng cửa
    df = data[['Close']].dropna()
    
    # 2. XỬ LÝ DỮ LIỆU & HUẤN LUYỆN LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.values)
    
    # Dùng 60 ngày trước để dự đoán ngày tiếp theo
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Chia tập Train (80%) và Test (20%)
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test = X[train_size:]
    
    if st.button("Chạy mô hình LSTM & Hiển thị biểu đồ"):
        with st.spinner("Mô hình LSTM đang tính toán xu hướng tăng giảm..."):
            # Khởi tạo mạng LSTM tối giản
            model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                LSTM(units=50),
                Dense(units=1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, batch_size=32, epochs=5, verbose=0) # Huấn luyện nhanh 5 vòng
            
            # Dự đoán trên tập kiểm tra (Test)
            predictions = model.predict(X_test)
            predictions = scaler.inverse_transform(predictions)
            
            # Tách dữ liệu thực tế và dữ liệu dự đoán để vẽ biểu đồ
            df_actual = df.iloc[60:]
            train_plot = df_actual.iloc[:train_size]
            test_plot = df_actual.iloc[train_size:].copy()
            test_plot['Predictions'] = predictions
            
            # 3. HIỂN THỊ BIỂU ĐỒ TĂNG GIẢM TƯƠNG TÁC
            fig = go.Figure()
            # Đường giá lịch sử dùng để học
            fig.add_trace(go.Scatter(x=train_plot.index, y=train_plot['Close'].values.flatten(), name='Giá Quá Khứ (Train)'))
            # Đường giá thực tế để đối chiếu
            fig.add_trace(go.Scatter(x=test_plot.index, y=test_plot['Close'].values.flatten(), name='Giá Thực Tế (Test)'))
            # Đường giá do mô hình LSTM đoán xu hướng
            fig.add_trace(go.Scatter(x=test_plot.index, y=test_plot['Predictions'].values.flatten(), name='Giá Dự Đoán (LSTM)', line=dict(dash='dash', color='red')))
            
            fig.update_layout(
                title=f"Biểu đồ Xu hướng Tăng Giảm mã {ticker}",
                xaxis_title="Thời gian",
                yaxis_title="Giá Cổ Phiếu",
                template="plotly_dark",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Dự đoán giá ngày mai
            last_60_days = scaled_data[-60:]
            X_tomorrow = np.array([last_60_days])
            X_tomorrow = np.reshape(X_tomorrow, (X_tomorrow.shape[0], X_tomorrow.shape[1], 1))
            tomorrow_pred = scaler.inverse_transform(model.predict(X_tomorrow))
            
            st.success(f"🔮 Giá đóng cửa dự đoán của ngày kế tiếp: **{tomorrow_pred[0][0]:,.2f}**")
