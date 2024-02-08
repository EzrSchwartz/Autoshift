from fitparse import FitFile
import pandas as pd
from math import radians, cos, sin, asin, sqrt


def process_files(files):
    gear_change_events = []
    records = []
    data = []  # List to store data for DataFrame
    last_shift_time = None
    files = find_valid_files(files)
    first_shift_processed = False
    for file in files:
        fitfile = FitFile(file)
        print(fitfile)
        last_record_distance = 0

        for message in fitfile.get_messages():
            if message.name == 'event' and message.get_value('event') in ('front_gear_change', 'rear_gear_change'):
                gear_change_events.append({
                    'timestamp': message.get_value('timestamp').timestamp(),
                    'rear_gear_num': message.get_value('rear_gear_num'),
                    'distance': last_record_distance  # Assuming distance is recorded in the previous record message
                })
            elif message.name == 'record':
                distance = message.get_value('distance')
                if distance is not None:
                    last_record_distance = distance

                record = {
                    'timestamp': message.get_value('timestamp').timestamp(),
                    'power': message.get_value('power'),
                    'cadence': message.get_value('cadence'),
                    'grade': message.get_value('grade'),
                    'distance': distance,
                    'speed':message.get_value('speed')
                }
                if all(record.get(field) is not None for field in ['power', 'cadence','grade','distance','speed']):
                    records.append(record)

    for event in gear_change_events:
        if not first_shift_processed:
            first_shift_processed = True
            last_gear = event['rear_gear_num']
            last_shift_time = event['timestamp']
            continue  # Skip the rest of the loop for the first event


        timestamp = event['timestamp']
        event_distance = event['distance']
        gear_num = event['rear_gear_num']

        # Find the record that matches the timestamp of the gear change event
        matching_record = next((record for record in records if record['timestamp'] == timestamp), None)
        if matching_record:
            # Determine shift direction (upshift or downshift)
            shift_direction = 'upshift' if gear_num > last_gear else 'downshift'

            # Calculate time since last shift
            time_since_last_shift = timestamp - last_shift_time if last_shift_time is not None else None
            last_shift_time = timestamp
            if time_since_last_shift >= 15:
                target_timestamp = timestamp - (time_since_last_shift / 2)
                stayrecord = next((record for record in records if record['timestamp'] == target_timestamp and all(
                    record.get(field) is not None for field in ['power', 'cadence', 'speed', 'grade'])), None)

                if stayrecord:
                    data.append({
                        'timestamp': target_timestamp,
                        'power': stayrecord['power'],
                        'cadence': stayrecord['cadence'],
                        'speed': stayrecord['speed'],
                        'average_grade': stayrecord['grade'],
                        'label': 0,
                        'time_since_last_shift': target_timestamp - matching_record['timestamp']
                    })

            # Calculate average grade over the next distance
            total_distance = 0
            total_grade = 0
            count = 0
            future_records = [record for record in records if timestamp <= record['timestamp'] < timestamp + 15]

            # Calculate average grade
            if future_records:
                total_grade = sum(record.get('grade')or 0 for record in future_records)
                average_grade = total_grade / len(future_records)
            else:
                average_grade = None

            speed = matching_record.get('speed')
            # speed = event['speed']e
            if(timestamp!=None and matching_record['power']!=None and matching_record['cadence']!= None and shift_direction!= None and time_since_last_shift!= None and average_grade!= None and speed != None):
                if shift_direction == "upshift":
                    shift_direction = 1
                if shift_direction == "downshift":
                    shift_direction = 2
                data.append({
                    'timestamp': timestamp,
                    'power': matching_record['power'],
                    'cadence': matching_record['cadence'],
                    'label': shift_direction,
                    'time_since_last_shift': time_since_last_shift,
                    'average_grade': average_grade,
                    'speed': speed
                })
            else:
                continue

            last_gear = event['rear_gear_num']
            last_shift_time = event['timestamp']


    df = pd.DataFrame(data)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(df)
    return distributeData(df)
def distributeData(df):
    return df['timestamp'], df['power'], df['cadence'], df['label'], df['time_since_last_shift'],df['average_grade'],df['speed']

def process_file(file_path):
    fitfile = FitFile(file_path)
    gear_change_events = [create_gear_change_event(message) for message in fitfile.get_messages() if is_gear_change_event(message)]
    records = [create_record(message) for message in fitfile.get_messages() if message.name == 'record']
    return gear_change_events, records

def is_gear_change_event(message):
    return message.name == 'event' and message.get_value('event') in ('front_gear_change', 'rear_gear_change')

def create_gear_change_event(message):
    return {
        'timestamp': message.get_value('timestamp').timestamp(),
        'rear_gear_num': message.get_value('rear_gear_num')
    }

def haversine(lat1, lon1, lat2, lon2):
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def create_record(message):
    speed_mps = message.get_value('speed')
    speed_mph = None
    if speed_mps is not None:
        speed_mph = speed_mps * 2.23694  # Convert from meters per second to miles per hour
    position_lat = message.get_value('position_lat')
    position_long = message.get_value('position_long')
    print(position_long)
    print(position_lat)
    record = {
        'timestamp': message.get_value('timestamp').timestamp(),
        'power': message.get_value('power'),
        'cadence': message.get_value('cadence'),
        'speed': speed_mph,  # Store speed in mph
        'grade': message.get_value('grade'),
        'position_lat': position_lat,
        'position_long': position_long
    }
    return record if all(value is not None for value in record.values()) else None

def has_all_required_fields(record):
    required_fields = ['power', 'cadence', 'speed', 'grade']
    return all(record.get(field) is not None for field in required_fields)


def is_relevant(record, event):
    # Add your logic to determine if a record is relevant for an event
    return abs(record['timestamp'] - event['timestamp']) <= 15


def find_valid_files(files):
    valid_files = []

    for file_path in files:
        print("Finding If Valid:", file_path)
        try:
            fitfile = FitFile(file_path)

            has_gear_shifts = False
            has_required_fields = True

            # Iterating through messages only once
            for message in fitfile.get_messages():
                # Check for gear shifts
                if not has_gear_shifts and is_gear_change_event(message):
                    has_gear_shifts = True

                # Check for required fields
                if message.name == 'record':
                    has_required_fields = has_all_required_fields(message)
                    if not has_required_fields:
                        continue  # Exit the loop early if a record is missing fields

            # Only add the file if both conditions are met
            if has_gear_shifts and has_required_fields:
                valid_files.append(file_path)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    return valid_files


def is_gear_change_event(message):
    return message.name == 'event' and message.get_value('event') in ('front_gear_change', 'rear_gear_change')

def has_all_required_fields(message):
    required_fields = ['power', 'cadence', 'speed', 'grade']
    return all(message.get_value(field) is not None for field in required_fields)

# Example usage


