import requests
import time
import schedule
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def connect_to_mongodb():
    """
    Establish a connection to MongoDB Atlas.
    
    Returns:
        db: MongoDB database object if successful, None otherwise.
    """
    try:
        client = MongoClient(
            "", # Replace with your MongoDB connection string
            serverSelectionTimeoutMS=5000
        )
        # Test the connection
        client.admin.command('ping')
        db = client["openaq_db"]
        logging.info("Successfully connected to MongoDB Atlas")
        return db
    except ConnectionFailure as e:
        logging.error(f"Error connecting to MongoDB: {e}")
        return None

def fetch_and_extract_sensors(api_key, location_id, sensor_id, name, db):
    """
    Fetch data from OpenAQ API and insert it into a MongoDB collection specific to the location_id.
    
    Args:
        api_key (str): The OpenAQ API key.
        location_id (int): The location ID to query.
        sensor_id (int): The sensor ID to match.
        name (str): The name of the measurement.
        db: MongoDB database object.
    
    Returns:
        bool: True if data was inserted, False otherwise.
    """
    url = f"https://api.openaq.org/v3/locations/{location_id}/latest"
    headers = {"X-API-Key": api_key}
    
    # Use a collection named after the location_id (e.g., location_1543132)
    collection = db[f"location_{location_id}"]
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if "results" in data and data["results"]:
                for result in data["results"]:
                    if sensor_id == result.get("sensorsId"):
                        values = result.get("value")
                        datetime = result.get("datetime")
                        if datetime:
                            document = {
                                "datetime_utc": datetime.get("utc"),
                                "datetime_local": datetime.get("local"),
                                "name": name,
                                "value": values
                            }
                            # Check for existing document to avoid duplicates
                            existing = collection.find_one({
                                "datetime_utc": document["datetime_utc"],
                                "name": document["name"]
                            })
                            if not existing:
                                collection.insert_one(document)
                                logging.info(f"Data inserted for Location ID {location_id}, Sensor ID {sensor_id} into collection location_{location_id}")
                                return True
                            else:
                                logging.info(f"Duplicate data skipped for Location ID {location_id}, Sensor ID {sensor_id}")
                                return False
                logging.info(f"No matching sensor ID {sensor_id} found for Location ID {location_id}.")
                return False
            else:
                logging.info(f"No measurements found for Location ID {location_id}.")
                return False
        else:
            logging.error(f"Error for Location ID {location_id}: Status code {response.status_code}. Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data for Location ID {location_id}: {e}")
        return False

def job():
    """
    Job to fetch data for all sensors and store in MongoDB.
    """
    db = connect_to_mongodb()
    if db is None:
        logging.error("Cannot proceed with job due to MongoDB connection failure")
        return

    # OpenAQ API key
    api_key = "" # Replace with your OpenAQ API key

    # Updated sensor details
    sensor_details = [
        {"location_id": 1543132, "sensor_id": 6909373, "name": "pm25 µg/m³", "location_name": "1543132_pm25"},
        {"location_id": 1563313, "sensor_id": 7971182, "name": "pm10 µg/m³", "location_name": "1563313_pm10"},
        {"location_id": 1563313, "sensor_id": 6939684, "name": "pm25 µg/m³", "location_name": "1563313_pm25"},
        {"location_id": 1563313, "sensor_id": 6939696, "name": "temperature c", "location_name": "1563313_temperature"},
        {"location_id": 2161296, "sensor_id": 7771983, "name": "co µg/m³", "location_name": "2161296_co"},
        {"location_id": 2161296, "sensor_id": 7772030, "name": "no2 µg/m³", "location_name": "2161296_no2"},
        {"location_id": 2161296, "sensor_id": 7772101, "name": "pm10 µg/m³", "location_name": "2161296_pm10"},
        {"location_id": 2161296, "sensor_id": 7772040, "name": "pm25 µg/m³", "location_name": "2161296_pm25"},
        {"location_id": 225579, "sensor_id": 1304368, "name": "pm25 µg/m³", "location_name": "225579_pm25"},
        {"location_id": 225631, "sensor_id": 1304293, "name": "pm25 µg/m³", "location_name": "225631_pm25"},
        {"location_id": 225661, "sensor_id": 1304390, "name": "co ppm", "location_name": "225661_co"},
        {"location_id": 225661, "sensor_id": 1304075, "name": "no2 ppm", "location_name": "225661_no2"},
        {"location_id": 225661, "sensor_id": 1304164, "name": "o3 ppm", "location_name": "225661_o3"},
        {"location_id": 225661, "sensor_id": 1304424, "name": "pm10 µg/m³", "location_name": "225661_pm10"},
        {"location_id": 225661, "sensor_id": 1304186, "name": "pm25 µg/m³", "location_name": "225661_pm25"},
        {"location_id": 2857820, "sensor_id": 9302673, "name": "pm10 µg/m³", "location_name": "2857820_pm10"},
        {"location_id": 2857820, "sensor_id": 9302670, "name": "pm25 µg/m³", "location_name": "2857820_pm25"},
        {"location_id": 2857820, "sensor_id": 9302700, "name": "relativehumidity %", "location_name": "2857820_relativehumidity"},
        {"location_id": 2857820, "sensor_id": 9302685, "name": "temperature c", "location_name": "2857820_temperature"},
        {"location_id": 2857820, "sensor_id": 9302660, "name": "um003 particles/cm³", "location_name": "2857820_um003"},
        {"location_id": 3038744, "sensor_id": 10429794, "name": "pm10 µg/m³", "location_name": "3038744_pm10"},
        {"location_id": 3038744, "sensor_id": 10429802, "name": "pm25 µg/m³", "location_name": "3038744_pm25"},
        {"location_id": 3038744, "sensor_id": 10429805, "name": "relativehumidity %", "location_name": "3038744_relativehumidity"},
        {"location_id": 3038744, "sensor_id": 10429793, "name": "temperature c", "location_name": "3038744_temperature"},
        {"location_id": 3038744, "sensor_id": 10429810, "name": "um003 particles/cm³", "location_name": "3038744_um003"},
        {"location_id": 3276359, "sensor_id": 11357395, "name": "pm10 µg/m³", "location_name": "3276359_pm10"},
        {"location_id": 3276359, "sensor_id": 11357424, "name": "pm25 µg/m³", "location_name": "3276359_pm25"},
        {"location_id": 3276359, "sensor_id": 11357398, "name": "relativehumidity %", "location_name": "3276359_relativehumidity"},
        {"location_id": 3276359, "sensor_id": 11357401, "name": "temperature c", "location_name": "3276359_temperature"},
        {"location_id": 3276359, "sensor_id": 11357394, "name": "um003 particles/cm³", "location_name": "3276359_um003"},
        {"location_id": 354124, "sensor_id": 2009811, "name": "co ppm", "location_name": "354124_co"},
        {"location_id": 354124, "sensor_id": 2009810, "name": "no2 ppm", "location_name": "354124_no2"},
        {"location_id": 354124, "sensor_id": 2009814, "name": "o3 ppm", "location_name": "354124_o3"},
        {"location_id": 354124, "sensor_id": 2009805, "name": "pm10 µg/m³", "location_name": "354124_pm10"},
        {"location_id": 354124, "sensor_id": 2009817, "name": "pm25 µg/m³", "location_name": "354124_pm25"}
    ]

    # Fetch and store data
    for sensor in sensor_details:
        location_id = sensor["location_id"]
        sensor_id = sensor["sensor_id"]
        name = sensor["name"]
        logging.info(f"Fetching data for Location ID {location_id} and Sensor ID {sensor_id}...")
        fetch_and_extract_sensors(api_key, location_id, sensor_id, name, db)
        time.sleep(0.3)  # Rate limiting for API requests

if __name__ == "__main__":
    # Run the job immediately
    logging.info("Running initial data fetch...")
    job()

    # Schedule the job to run every hour
    schedule.every(1).hours.do(job)
    
    logging.info("Starting scheduler to fetch data every hour...")
    
    # Run the scheduler
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute for pending jobs
        except Exception as e:
            logging.error(f"Error in scheduler: {e}")
            time.sleep(300)  # Wait 5 minutes before retrying