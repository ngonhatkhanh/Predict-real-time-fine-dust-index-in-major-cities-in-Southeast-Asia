from fastapi import FastAPI
from pymongo import MongoClient
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Cấu hình CORS để cho phép Streamlit truy cập API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dữ liệu vị trí
location_data = [
    {"name": "โรงเรียนเตรียมทหาร รุ่นที่ 13 อนุสรณ์", "location_id": 2857820, "coordinates": {"lat": 16.350857980092687, "lng": 104.57883941275776}, "Country": "Thailand"},
    {"name": "CMT8", "location_id": 3276359, "coordinates": {"lat": 10.78533, "lng": 106.67029}, "Country": "Vietnam"},
    {"name": "Yupparaj Wittayalai School", "location_id": 225579, "coordinates": {"lat": 18.7909333, "lng": 98.99}, "Country": "Thailand"},
    {"name": "Đào Duy Từ", "location_id": 2161296, "coordinates": {"lat": 21.0354, "lng": 105.8529}, "Country": "Vietnam"},
    {"name": "Qoryah Darussalam", "location_id": 1563313, "coordinates": {"lat": -6.3612408, "lng": 106.8419476}, "Country": "Indonesia"},
    {"name": "Ocean Park", "location_id": 3038744, "coordinates": {"lat": 1.311493, "lng": 103.928128}, "Country": "Singapore"},
    {"name": "Manila", "location_id": 1543132, "coordinates": {"lat": 14.57711, "lng": 120.9778}, "Country": "Philippines"},
    {"name": "Nong Thin Public Park, Nong Khai", "location_id": 225661, "coordinates": {"lat": 17.877479999999995, "lng": 102.728925}, "Country": "Thailand"},
    {"name": "National Economic and Social Development Council Office", "location_id": 225631, "coordinates": {"lat": 13.756281, "lng": 100.514267}, "Country": "Thailand"}
]

# Danh sách cột
columns = [
    '1543132_pm25', '1563313_pm25', '1563313_temperature',
    '2161296_co', '2161296_no2', '2161296_pm10', '2161296_pm25',
    '225579_pm25', '225631_pm25', '225661_co', '225661_no2', '225661_o3',
    '225661_pm10', '225661_pm25', '2857820_pm25',
    '2857820_relativehumidity', '2857820_temperature', '2857820_um003',
    '3038744_pm25', '3038744_relativehumidity', '3038744_temperature',
    '3038744_um003', '3276359_pm25', '3276359_relativehumidity',
    '3276359_temperature', '3276359_um003',
]

def get_location_id(column):
    return column.split('_')[0]

def get_param(column):
    return column.split('_')[1]

def fetch_data_from_mongodb():
    client = MongoClient('') #Your mongoDB connection string here
    db = client['openaq_db']
    
    data = {col: [] for col in columns}
    
    for col in columns:
        location_id = get_location_id(col)
        param = get_param(col)
        collection = db[f'location_{location_id}']
        
        cursor = collection.find(
            {'name': {'$regex': param, '$options': 'i'}}
        ).sort('datetime_utc', -1).limit(336)
        
        values = [doc.get('value', None) for doc in cursor]
        values = values[::-1]  # Đảo ngược để từ cũ nhất đến mới nhất
        data[col] = values
    
    df = pd.DataFrame(data)
    client.close()
    
    # Chuyển đổi dữ liệu thành định dạng dài (long format)
    long_df = pd.DataFrame()
    for col in columns:
        location_id = get_location_id(col)
        param = get_param(col)
        temp_df = pd.DataFrame({
            'location_id': [location_id] * len(df[col]),
            'parameter': [param] * len(df[col]),
            'value': df[col],
            'time_index': range(len(df[col]))
        })
        long_df = pd.concat([long_df, temp_df], ignore_index=True)
    
    # Ánh xạ với thông tin vị trí
    location_df = pd.DataFrame(location_data).rename(columns={'name': 'location_name'})
    long_df['location_id'] = long_df['location_id'].astype(int)
    location_df['location_id'] = location_df['location_id'].astype(int)
    long_df = long_df.merge(location_df[['location_id', 'location_name', 'Country', 'coordinates']], on='location_id', how='left')
    
    # Thêm cột lat, lng
    long_df['lat'] = long_df['coordinates'].apply(lambda x: x['lat'] if pd.notnull(x) else None)
    long_df['lng'] = long_df['coordinates'].apply(lambda x: x['lng'] if pd.notnull(x) else None)
    long_df = long_df.drop(columns=['coordinates'])
    
    return long_df

@app.get("/data")
async def get_data():
    df = fetch_data_from_mongodb()
    return df.to_dict(orient="records")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)