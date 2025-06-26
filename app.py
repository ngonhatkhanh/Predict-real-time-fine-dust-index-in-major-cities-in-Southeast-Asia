import pandas as pd
import plotly.express as px
import streamlit as st
import requests
import torch
import sys
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime, timedelta

# Đặt cấu hình trang ngay đầu script
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")

# Thêm CSS cho recommendation-box
st.markdown(
    """
    <style>
    .recommendation-box {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Thêm thư mục SOFTS vào sys.path (thay đường dẫn này bằng đường dẫn thực tế trên máy của bạn)
SOFTS_PATH = r"C:\Users\NITRO\Desktop\Workspace\CS313\AQI-Globe\SOFTS"
sys.path.append(SOFTS_PATH)

# Kiểm tra xem có thể import Exp_Custom không
try:
    from exp.exp_custom import Exp_Custom
except ModuleNotFoundError as e:
    st.error(f"Không thể import Exp_Custom. Hãy đảm bảo thư mục 'SOFTS' tồn tại tại {SOFTS_PATH} và chứa 'exp/exp_custom.py'. Lỗi: {e}")
    st.stop()

# Định nghĩa các lớp mô hình
class STAR(nn.Module):
    def __init__(self, d_series, d_core):
        super(STAR, self).__init__()
        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, input, *args, **kwargs):
        batch_size, channels, d_series = input.shape
        combined_mean = F.gelu(self.gen1(input))
        combined_mean = self.gen2(combined_mean)
        weight = F.softmax(combined_mean, dim=1)
        combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True).repeat(1, channels, 1)
        combined_mean_cat = torch.cat([input, combined_mean], -1)
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat))
        combined_mean_cat = self.gen4(combined_mean_cat)
        return combined_mean_cat, None

class DataEmbedding_inverted(nn.Module):
    def __init__(self, seq_len, d_model, dropout=0.05):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark=None):
        x = x.permute(0, 2, 1)
        x = self.value_embedding(x)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=512, dropout=0.05, activation="gelu"):
        super(EncoderLayer, self).__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x, attn_mask=None):
        new_x, _ = self.attention(x)
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        x = x.permute(0, 2, 1)
        y = self.conv1(x)
        y = self.activation(y)
        y = self.conv2(y)
        y = y.permute(0, 2, 1)
        y = self.norm2(x.permute(0, 2, 1) + self.dropout(y))
        return y, None

class Encoder(nn.Module):
    def __init__(self, attn_layers):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)

    def forward(self, x, attn_mask=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask)
            attns.append(attn)
        return x, attns

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)
        self.use_norm = configs.use_norm
        self.encoder = Encoder(
            [
                EncoderLayer(
                    STAR(configs.d_model, configs.d_core),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.e_layers)
            ]
        )
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]

# Hàm phân loại AQI và trả về khuyến nghị 
def classify_aqi(value):
    if value <= 50:
        return {
            "advice": "Không khí trong lành. Bạn có thể hoạt động ngoài trời bình thường."
        }
    elif value <= 100:
        return {
            "advice": "Chất lượng không khí có thể chấp nhận được tuy nhiên, đối với người có da nhạy cảm nên hạn chế ra ngoài trong thời gian dài."
        }
    elif value <= 150:
        return {
            "advice": "Người có bệnh hô hấp, người già, trẻ nhỏ nên hạn chế hoạt động ngoài trời trong thời gian dài."
        }
    elif value <= 200:
        return {
            "advice": "Bạn nên hạn chế hoạt động ngoài trời. Đối với những người có da nhạy cảm nên ở trong nhà trong thời gian này."
        }
    elif value <= 300:
        return {
            "advice": "Bạn nên hạn chế tối đa các hoạt động ngoài trời. Đối với nhóm người có da nhạy cảm nên tránh ra ngoài hoàn toàn."
        }
    else:
        return {
            "advice": "Tình trạng ô nhiễm nghiêm trọng, bạn nên ở trong nhà. Nếu ra ngoài trong thời gian dài có thể gây nguy hiểm đến tính mạng."
        }

# Hàm tính khoảng cách Haversine giữa hai điểm (đơn vị: km)
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Bán kính Trái Đất (km)
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Hàm tính AQI cho từng chất gây ô nhiễm (theo chuẩn VN_AQI)
def calculate_pollutant_aqi(pollutant, value):
    # Ngưỡng dựa trên VN_AQI (tham khảo từ tài liệu, điều chỉnh cho phù hợp)
    ranges = {
        'pm25': [
            (0, 12.0, 0, 50),    # Tốt
            (12.1, 35.4, 51, 100),  # Trung bình
            (35.5, 55.4, 101, 150),  # Kém
            (55.5, 150.4, 151, 200),  # Xấu
            (150.5, 250.4, 201, 300),  # Rất xấu
            (250.5, 500.4, 301, 500)  # Nguy hại
        ],
        'pm10': [
            (0, 54, 0, 50),
            (55, 154, 51, 100),
            (155, 254, 101, 150),
            (255, 354, 151, 200),
            (355, 424, 201, 300),
            (425, 604, 301, 500)
        ],
        'no2': [  # µg/m³
            (0, 40, 0, 50),
            (41, 80, 51, 100),
            (81, 180, 101, 150),
            (181, 280, 151, 200),
            (281, 400, 201, 300),
            (401, 750, 301, 500)
        ],
        'co': [  # mg/m³
            (0, 4.4, 0, 50),
            (4.5, 9.4, 51, 100),
            (9.5, 12.4, 101, 150),
            (12.5, 15.4, 151, 200),
            (15.5, 30.4, 201, 300),
            (30.5, 50.4, 301, 500)
        ],
        'o3': [  # µg/m³ (8 giờ trung bình)
            (0, 50, 0, 50),
            (51, 100, 51, 100),
            (101, 140, 101, 150),
            (141, 200, 151, 200),
            (201, 300, 201, 300),
            (301, 500, 301, 500)
        ]
    }
    pollutant_ranges = ranges.get(pollutant, [(0, 500, 0, 500)])
    if pd.isna(value) or value is None:
        return 0  # Mặc định AQI = 0 nếu thiếu dữ liệu
    for c_low, c_high, i_low, i_high in pollutant_ranges:
        if c_low <= value <= c_high:
            return round(((i_high - i_low) / (c_high - c_low)) * (value - c_low) + i_low)
    return 0

# Hàm tính AQI tổng quát
def calculate_aqi(pollutants):
    aqi_values = []
    for pollutant, value in pollutants.items():
        if pd.notna(value) and value is not None and value > 0:
            aqi = calculate_pollutant_aqi(pollutant, value)
            aqi_values.append(aqi)
    return max(aqi_values) if aqi_values else 0

# Hàm nhóm các trạm và chia sẻ chỉ số
def share_pollutants(data, distance_threshold=50):
    pollutants = data['parameter'].dropna().unique()
    pollutants = [p for p in pollutants if p in ['pm25', 'pm10', 'no2', 'co', 'o3']]
    
    stations = data[['location_id', 'lat', 'lng']].drop_duplicates()
    
    groups = []
    used = set()
    for i, station in stations.iterrows():
        if station['location_id'] in used:
            continue
        group = [station['location_id']]
        used.add(station['location_id'])
        for j, other_station in stations.iterrows():
            if other_station['location_id'] in used or other_station['location_id'] == station['location_id']:
                continue
            dist = haversine_distance(station['lat'], station['lng'], other_station['lat'], other_station['lng'])
            if dist <= distance_threshold:
                group.append(other_station['location_id'])
                used.add(other_station['location_id'])
        groups.append(group)
    
    latest_data = data.sort_values('time_index').groupby(['location_id', 'parameter'])['value'].last().unstack().reset_index()
    
    updated_data = data.copy()
    for group in groups:
        group_data = latest_data[latest_data['location_id'].isin(group)]
        for pollutant in pollutants:
            if pollutant not in group_data.columns:
                continue
            missing_stations = group_data['location_id'][group_data[pollutant].isna()]
            if not missing_stations.empty:
                valid_values = group_data[pollutant].dropna()
                if not valid_values.empty:
                    mean_value = valid_values.mean()
                    for station in missing_stations:
                        mask = (updated_data['location_id'] == station) & (updated_data['parameter'] == pollutant)
                        latest_time = updated_data[mask]['time_index'].max() if mask.any() else None
                        if pd.notna(latest_time):
                            if not mask.any():
                                new_row = updated_data[updated_data['location_id'] == station].iloc[0].copy()
                                new_row['parameter'] = pollutant
                                new_row['value'] = mean_value
                                new_row['time_index'] = latest_time
                                updated_data = pd.concat([updated_data, pd.DataFrame([new_row])], ignore_index=True)
                            else:
                                updated_data.loc[mask & (updated_data['time_index'] == latest_time), 'value'] = mean_value
    
    return updated_data

# Hàm tải mô hình
def load_model(model_path=r"C:\Users\NITRO\Desktop\Workspace\CS313\AQI-Globe\softs.pt"):
    import argparse
    config = {
        'root_path': './',
        'data_path': '/kaggle/input/openaq-26-sensors/merged_openaq_2024_2025_timeseries_cleaned.csv',
        'data': 'air_pm25',
        'features': 'M',
        'freq': 'h',
        'seq_len': 336,
        'pred_len': 24,
        'model': 'SOFTS',
        'checkpoints': './checkpoints/',
        'd_model': 256,
        'd_core': 128,
        'd_ff': 512,
        'e_layers': 4,
        'learning_rate': 0.00005,
        'lradj': 'cosine',
        'train_epochs': 100,
        'patience': 10,
        'batch_size': 8,
        'dropout': 0.05,
        'activation': 'gelu',
        'use_norm': True,
        'num_workers': 0,
        'use_gpu': True,
        'gpu': '0',
        'save_model': True,
    }
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    args.__dict__.update(config)
    args.use_gpu = torch.cuda.is_available() and args.use_gpu
    try:
        exp = Exp_Custom(args)
        exp.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        exp.model.eval()
        return exp
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}")
        return None

# Hàm dự đoán cho tất cả các chỉ số
def predict_pollutants(exp, data, location_id, hours_ahead, pollutants=['pm25', 'pm10', 'no2', 'co', 'o3']):
    predictions = {}
    for pollutant in pollutants:
        pollutant_data = data[(data['location_id'] == location_id) & (data['parameter'] == pollutant)]
        if pollutant_data.empty or len(pollutant_data) < 336:
            predictions[pollutant] = 0  # Đặt mặc định là 0 nếu thiếu dữ liệu
            continue
        
        pollutant_data = pollutant_data.sort_values('time_index', ascending=True)
        input_data = pollutant_data['value'].tail(336).values
        
        input_tensor = torch.tensor(input_data, dtype=torch.float32).reshape(1, 336, 1)
        x_mark_enc = torch.zeros(1, 336, 1)
        
        try:
            with torch.no_grad():
                output = exp.model(input_tensor, x_mark_enc, None, None)
                if hours_ahead <= 24:
                    predicted_value = output[0, min(hours_ahead - 1, 23), 0].item()
                    predictions[pollutant] = predicted_value
                else:
                    predicted_values = output[0, :, 0].numpy().tolist()
                    predictions[pollutant] = predicted_values[0]  # Lấy giá trị đầu tiên
        except Exception as e:
            predictions[pollutant] = 0  # Đặt mặc định là 0 nếu xảy ra lỗi
    
    return predictions

# Hàm dự đoán AQI cho 24 giờ tiếp theo từ thời gian hiện tại
def predict_daily_aqi(exp, data, location_id, current_time, pollutants=['pm25', 'pm10', 'no2', 'co', 'o3']):
    daily_predictions = []
    # Tính số giờ còn lại đến 07:00 AM ngày hôm sau
    end_time = current_time.replace(hour=7, minute=0, second=0, microsecond=0) + timedelta(days=1)
    if current_time.hour >= 7:
        end_time += timedelta(days=1)
    hours_to_end = int((end_time - current_time).total_seconds() / 3600)

    # Dự đoán AQI cho từng giờ từ hiện tại đến 07:00 AM ngày hôm sau
    for hour in range(1, hours_to_end + 1):
        predictions = predict_pollutants(exp, data, location_id, hour)
        latest_pollutants = {p: predictions.get(p, 0) for p in pollutants}
        aqi = calculate_aqi(latest_pollutants)
        daily_predictions.append(aqi)
    
    return daily_predictions

# Hàm tìm khoảng thời gian tốt nhất (AQI ≤ 100: Tốt và Trung bình)
def find_best_time_slots(aqi_values, current_time):
    if not aqi_values:
        return None, None, False, False
    best_slots = []
    start = None
    
    for i, aqi in enumerate(aqi_values):
        if aqi <= 100 and start is None:
            start = i
        elif aqi > 100 and start is not None:
            end = i - 1
            best_slots.append((start, end, end - start + 1))
            start = None
    if start is not None:
        end = len(aqi_values) - 1
        best_slots.append((start, end, end - start + 1))
    
    if not best_slots:
        return None, None, False, False
    
    # Chọn khoảng thời gian dài nhất hoặc sớm nhất nếu có nhiều khoảng bằng nhau
    best_slot = max(best_slots, key=lambda x: (x[2], -x[0]))  # Ưu tiên độ dài, rồi ưu tiên giờ sớm
    start_hour = (current_time + timedelta(hours=best_slot[0] + 1)).hour
    end_hour = (current_time + timedelta(hours=best_slot[1] + 1)).hour
    
    # Xác định ngày của start và end
    start_date = (current_time + timedelta(hours=best_slot[0] + 1)).date()
    end_date = (current_time + timedelta(hours=best_slot[1] + 1)).date()
    current_date = current_time.date()
    
    return start_hour, end_hour, start_date == current_date, end_date == current_date

# Hàm lấy dữ liệu từ API
def fetch_data_from_api():
    try:
        response = requests.get("http://localhost:8000/data")
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        return df
    except requests.RequestException as e:
        st.error(f"Lỗi khi lấy dữ liệu từ API: {e}")
        return pd.DataFrame()

# Lấy dữ liệu
final = fetch_data_from_api()

# Kiểm tra xem dữ liệu có rỗng không
if final.empty:
    st.error("Không thể tải dữ liệu. Vui lòng kiểm tra API.")
    st.stop()

# Đảm bảo location_id là kiểu int
final['location_id'] = final['location_id'].astype(int)

# Chia sẻ chỉ số giữa các trạm gần nhau
final = share_pollutants(final, distance_threshold=50)

# Hàm tính AQI và phân loại
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Tốt"
    elif aqi <= 100:
        return "Trung bình"
    elif aqi <= 150:
        return "Kém"
    elif aqi <= 200:
        return "Xấu"
    elif aqi <= 300:
        return "Rất xấu"
    else:
        return "Nguy hại"

# Tính AQI cho từng địa điểm
pollutants = final['parameter'].dropna().unique()
pollutants = [p for p in pollutants if p in ['pm25', 'pm10', 'no2', 'co', 'o3']]
latest_data = final.sort_values('time_index').groupby(['location_id', 'location_name', 'Country', 'lat', 'lng', 'parameter'])['value'].last().unstack().reset_index()
latest_data['aqi'] = latest_data.apply(
    lambda row: calculate_aqi({p: row[p] for p in pollutants if p in row and pd.notna(row[p])}), axis=1
)
latest_data['category'] = latest_data['aqi'].apply(get_aqi_category)

# Thêm cột aqi_text để hiển thị giá trị AQI trên bản đồ
latest_data['aqi_text'] = latest_data['aqi'].astype(str)

# Tạo cột size để làm nổi bật location được chọn
latest_data['size'] = 10

# Định nghĩa thang màu AQI đầy đủ
aqi_colors = {
    "Tốt": "green",
    "Trung bình": "yellow",
    "Kém": "orange",
    "Xấu": "red",
    "Rất xấu": "purple",
    "Nguy hại": "maroon"
}

# Thời gian hiện tại
current_time = datetime.now()  # Lấy thời gian hệ thống

st.title("🌍 Air Quality Monitoring Dashboard")

# Hiển thị các tab (Ranking, Map, Predict) cùng hàng
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.subheader(" Xếp hạng trực tiếp địa điểm sạch nhất theo thời gian thực")
    # Tạo bảng xếp hạng với cột Thứ hạng và Địa điểm
    ranking_data = latest_data[['location_name', 'aqi']].sort_values('aqi')
    # Thêm cột Thứ hạng (Rank) từ 1 trở lên
    ranking_data['Thứ hạng'] = range(1, len(ranking_data) + 1)
    # Sắp xếp lại cột để Thứ hạng nằm trước Địa điểm
    ranking_data = ranking_data[['Thứ hạng', 'location_name']].rename(columns={'location_name': 'Địa điểm'})
    # Reset index và loại bỏ cột index
    ranking_data = ranking_data.reset_index(drop=True)
    # Sử dụng st.table() hoặc st.dataframe() với hide_index
    st.dataframe(ranking_data, hide_index=True, use_container_width=True)

with col2:
    st.subheader("🗺️ Bản đồ chất lượng không khí (AQI)")
    # Dropdown để chọn location làm nổi bật trên bản đồ
    unique_locations = latest_data['location_name'].unique()
    highlight_location = st.selectbox('Chọn địa điểm để làm nổi bật trên bản đồ:', ['Tất cả'] + list(unique_locations))

    if highlight_location != 'Tất cả':
        latest_data.loc[latest_data['location_name'] == highlight_location, 'size'] = 80

    # Đảm bảo hiển thị toàn bộ thang đo AQI trong legend
    fig_map = px.scatter_mapbox(
        latest_data,
        lat='lat',
        lon='lng',
        size='size',
        color='category',
        color_discrete_map=aqi_colors,
        category_orders={"category": ["Tốt", "Trung bình", "Kém", "Xấu", "Rất xấu", "Nguy hại"]},
        text='aqi_text',  # Hiển thị giá trị AQI trên marker
        hover_name='location_name',
        hover_data={'category': True, 'aqi': True, 'lat': False, 'lng': False, 'size': False},
        mapbox_style='carto-positron',
        zoom=2,
        title="AQI Across Locations"
    )
    fig_map.update_traces(
        textposition='middle center',  # Đặt vị trí text ở giữa marker
        textfont=dict(size=12, color='black'),  # Tùy chỉnh font chữ của AQI
        marker=dict(sizemin=10)  # Đảm bảo marker đủ lớn để hiển thị text
    )
    fig_map.update_layout(
        legend_title_text='AQI',
        legend=dict(
            title="Mức AQI",
            itemsizing='constant',
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='right',
            x=1
        )
    )
    st.plotly_chart(fig_map, use_container_width=True)

with col3:
    st.subheader("Dự đoán")
    # Tạo tab cho AQI và Sức khỏe
    tab1, tab2 = st.tabs(["AQI", "Sức khỏe"])

    # Tab AQI
    with tab1:
        st.subheader("🌫️ Dự đoán AQI")
        if highlight_location != 'Tất cả':
            location_id = latest_data[latest_data['location_name'] == highlight_location]['location_id'].iloc[0]
            current_aqi = latest_data[latest_data['location_name'] == highlight_location]['aqi'].iloc[0]

            # Tải mô hình
            exp = load_model(r"C:\Users\NITRO\Desktop\Workspace\CS313\AQI-Globe\softs.pt")

            # Hiển thị AQI hiện tại
            if current_aqi is not None and pd.notna(current_aqi):
                st.write(f"**AQI Hiện tại**: {current_aqi}")
                classification = classify_aqi(current_aqi)
                st.markdown(
                    f"""
                    <div class="recommendation-box">
                        {classification['advice']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                # Thêm khuyến nghị giờ hoạt động
                if exp is not None:
                    daily_aqi = predict_daily_aqi(exp, final, location_id, current_time)
                    if daily_aqi and all(aqi <= 100 for aqi in daily_aqi):
                        st.markdown(
                            f"""
                            <div class="recommendation-box">
                                Bạn có thể tận hưởng cả ngày hôm nay ở ngoài trời.
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        start_hour, end_hour, start_is_today, end_is_today = find_best_time_slots(daily_aqi, current_time)
                        if start_hour is not None and end_hour is not None:
                            if start_hour == end_hour:
                                if start_is_today:
                                    st.markdown(
                                        f"""
                                        <div class="recommendation-box">
                                            Thời điểm tốt nhất để hoạt động ngoài trời: {start_hour:02d}h hôm nay.
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.markdown(
                                        f"""
                                        <div class="recommendation-box">
                                            Thời điểm tốt nhất để hoạt động ngoài trời: {start_hour:02d}h ngày mai.
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                            elif start_is_today and end_is_today:
                                st.markdown(
                                    f"""
                                    <div class="recommendation-box">
                                        Thời điểm tốt nhất để hoạt động ngoài trời: {start_hour:02d}-{end_hour:02d}h hôm nay.
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            elif start_is_today:
                                st.markdown(
                                    f"""
                                    <div class="recommendation-box">
                                        Thời điểm tốt nhất để hoạt động ngoài trời: {start_hour:02d}h hôm nay - {end_hour:02d}h ngày mai.
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                    f"""
                                    <div class="recommendation-box">
                                        Không có thời điểm tốt nào để hoạt động ngoài trời hôm nay.
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                        else:
                            st.markdown(
                                f"""
                                <div class="recommendation-box">
                                    Không có thời điểm tốt nào để hoạt động ngoài trời hôm nay.
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
            else:
                st.write("**AQI Hiện tại**: Không có dữ liệu")

            # Nhập số giờ để dự đoán và hiển thị kết quả dự đoán
            hours_ahead = st.number_input("🕒 Nhập số giờ muốn dự đoán:", min_value=1, max_value=24, value=1, step=1, key="hours_input")
            if st.button("Dự đoán"):
                predicted_values = predict_pollutants(exp, final, location_id, hours_ahead)
                if predicted_values is not None:
                    latest_pollutants = latest_data[latest_data['location_id'] == location_id].iloc[0].to_dict()
                    latest_pollutants = {k: v for k, v in latest_pollutants.items() if k in pollutants and pd.notna(v)}
                    for pollutant in pollutants:
                        if pollutant in predicted_values and predicted_values[pollutant] is not None and predicted_values[pollutant] > 0:
                            latest_pollutants[pollutant] = predicted_values[pollutant]
                        else:
                            latest_pollutants[pollutant] = 0
                    predicted_aqi_hour = calculate_aqi(latest_pollutants)
                    valid_pollutants = [p.upper() for p in pollutants if latest_pollutants[p] > 0]
                    if valid_pollutants:
                        pollutant_str = ", ".join(valid_pollutants)
                        category = get_aqi_category(predicted_aqi_hour) if predicted_aqi_hour is not None else "Không xác định"
                        st.success(f"Dự đoán chỉ số AQI (dựa trên {pollutant_str}) sau {hours_ahead} giờ nữa: {predicted_aqi_hour} - Mức độ: {category}")
                    else:
                        st.warning(f"Không có dữ liệu dự đoán cho bất kỳ chỉ số nào tại địa điểm này sau {hours_ahead} giờ.")
                    if predicted_aqi_hour is not None and predicted_aqi_hour > 0:
                        classification = classify_aqi(predicted_aqi_hour)
                        st.markdown(
                            f"""
                            <div class="recommendation-box">
                                Trong {hours_ahead} giờ nữa:<br>
                                {classification['advice']}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

    # Tab Sức khỏe
    with tab2:
        st.subheader("🩺 Dự đoán Sức khỏe Cá nhân")
        if highlight_location != 'Tất cả':
            location_id = latest_data[latest_data['location_name'] == highlight_location]['location_id'].iloc[0]
            # Lấy giá trị PM2.5 trực tiếp từ cột 'pm25'
            current_pm25 = latest_data[latest_data['location_id'] == location_id]['pm25'].iloc[0] if not latest_data[latest_data['location_id'] == location_id].empty else 0

            # Input thông tin người dùng
            st.write("ℹ️ **Nhập thông tin cá nhân:**")
            age = st.number_input("🧍‍♂️ Tuổi", min_value=0, max_value=120, step=1)
            health_condition = st.selectbox("🫁 Tình trạng sức khỏe nền", ["Không có", "Bệnh liên quan đến hô hấp", "Bệnh liên quan đến tim mạch", "Cả hai"])
            outdoor_time = st.number_input("🕐 Thời gian ngoài trời (giờ)", min_value=0.0, max_value=24.0, step=0.5)
            activity_level = st.selectbox("🧭 Hoạt động ngoài trời", ["Nghỉ ngơi", "Vận động nhẹ (đi dạo)", "Vận động vừa (đạp xe)", "Vận động nặng (chạy bộ)"])
            wears_mask = st.checkbox("😷 Có đeo khẩu trang không?", value=False)

            # Thêm nút Dự đoán
            if st.button("Dự đoán", key="health_predict"):
                # Tính Exposure Factor
                # 1. duration_factor = hours_outside
                duration_factor = outdoor_time

                # 2. activity_multiplier
                activity_mapping = {
                    "Nghỉ ngơi": 1.0,
                    "Vận động nhẹ (đi dạo)": 1.2,
                    "Vận động vừa (đạp xe)": 1.5,
                    "Vận động nặng (chạy bộ)": 2.0
                }
                activity_multiplier = activity_mapping[activity_level]

                # 3. mask_efficiency
                mask_efficiency = 0.5 if wears_mask else 1.0

                # Tổng Exposure Factor
                exposure_factor = duration_factor * activity_multiplier * mask_efficiency

                # Tính Sensitivity Factor
                sensitivity_factor = 1.0  # Giá trị cơ bản
                # Điều chỉnh theo tuổi
                if age < 12 or age > 65:
                    sensitivity_factor += 0.5
                # Điều chỉnh theo bệnh nền
                health_condition_mapping = {
                    "Không có": 0,
                    "Bệnh liên quan đến hô hấp": 0.3,
                    "Bệnh liên quan đến tim mạch": 0.3,
                    "Cả hai": 0.6
                }
                num_conditions = health_condition_mapping[health_condition]
                sensitivity_factor += num_conditions

                # Giá trị Potency (hằng số)
                potency = 0.006

                # Tính Risk Score: Risk Score = PM2.5 × Exposure Factor × Sensitivity Factor × Potency
                risk_score = current_pm25 * exposure_factor * sensitivity_factor * potency

                # Đánh giá Risk Score (theo ngưỡng mới)
                risk_description = ""
                risk_icon = ""
                if risk_score < 0.03:
                    risk_description = "An toàn, sức khỏe bình thường"
                    risk_icon = "✅"
                elif 0.03 <= risk_score <= 0.06:
                    risk_description = "Hạn chế ra ngoài nếu không cần thiết"
                    risk_icon = "⚠️"
                elif 0.06 < risk_score <= 0.1:
                    risk_description = "Khuyến nghị tránh ra ngoài trong thời gian dài, cân nhắc ở trong nhà"
                    risk_icon = "🚫"
                else:  # risk_score > 0.1
                    risk_description = "Cảnh báo nghiêm trọng, ở trong nhà và tìm kiếm tư vấn bác sĩ"
                    risk_icon = "🛑"

                st.markdown(
                    f"""
                    <div class="recommendation-box">
                        {risk_icon} {risk_description}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.write("Vui lòng chọn một địa điểm để dự đoán sức khỏe.")

# Hiển thị biểu đồ đường
st.subheader("📉 Biểu đồ")
location_selected = st.selectbox('Chọn địa điểm:', unique_locations)

temp_df = final[final['location_name'] == location_selected].copy()
line_df = temp_df.pivot(index='time_index', columns='parameter', values='value').reset_index()
line_df['Time'] = pd.date_range(start='2025-05-01', periods=len(line_df), freq='H')
line_df_melt = line_df.melt(id_vars=['Time'], value_vars=[col for col in line_df.columns if col != 'Time' and col != 'time_index'],
                            var_name='Parameter', value_name='Value')

value_range = line_df_melt['Value'].dropna()
if not value_range.empty:
    min_val, max_val = value_range.min(), value_range.max()
    tickvals = list(range(int(min_val), int(max_val) + 1, max(1, int((max_val - min_val) / 10))))
else:
    tickvals = [0]

fig_line = px.line(
    line_df_melt,
    x='Time',
    y='Value',
    color='Parameter',
    title=f"Air Quality Parameters at {location_selected}"
)
fig_line.update_layout(
    yaxis=dict(
        tickvals=tickvals,
        title="Giá trị",
        gridcolor='lightgray'
    ),
    xaxis_title="Thời gian",
    showlegend=True
)
st.plotly_chart(fig_line, use_container_width=True)