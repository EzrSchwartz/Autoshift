from fitparse import FitFile
import pandas as pd
from fitparse import FitFile
import pandas as pd

from fitparse import FitFile
import pandas as pd

def process_files(files, time_threshold=30):
    data_list = []
    gear_shift_counts = {'up': 0, 'down': 0}

    for file_path in files:
        print(f"Processing file: {file_path}")
        gear_change_events = sum(map(process_file, [file_path]), [])
        records = sum(map(extract_records, [file_path]), [])

        gear_change_events = gear_change_events[1:]  # Skip the first event
        last_gearchange = gear_change_events[2:]

        last_gear = None
        last_event_timestamp = None
        for event in gear_change_events:
            try:
                if last_event_timestamp is not None and event['timestamp'] - last_event_timestamp <= time_threshold:
                    event['label'] = determine_label(event['rear_gear_num'], last_gear)
                    gear_shift_counts = update_gear_shift_counts(event['label'], gear_shift_counts)
                else:
                    event['label'] = 2  # No gear shift

                last_gear = event['rear_gear_num']
                last_event_timestamp = event['timestamp']
                data_point = process_event(event, records, last_gearchange)
                if data_point:
                    data_list.append(data_point)
            except (ValueError, TypeError) as e:
                print(f"Skipping due to error: {e}")

        not_shifting_data_points = create_not_shifting_data_points(records, last_gearchange, time_threshold)
        data_list.extend(not_shifting_data_points)

    df = pd.DataFrame(data_list) if data_list else pd.DataFrame(columns=['power', 'cadence', 'speed', 'grade', 'label'])
    return df, gear_shift_counts

# Rest of your functions (process_file, extract_records, etc.) remain the same

def update_gear_shift_counts(label, counts):
    if label == 0:
        counts['down'] += 1
    elif label == 1:
        counts['up'] += 1
    return counts
# Rest of your code remains the same

def process_file(file_path):
    fitfile = FitFile(file_path)
    gear_change_events = [create_gear_change_event(message) for message in fitfile.get_messages() if is_gear_change_event(message)]
    return gear_change_events

def extract_records(file_path):
    fitfile = FitFile(file_path)
    return [create_record(message) for message in fitfile.get_messages() if message.name == 'record']

def create_gear_change_event(message):
    rear_gear_num = message.get_value('rear_gear_num')
    timestamp = message.get_value('timestamp').timestamp()
    return {'timestamp': timestamp, 'rear_gear_num': rear_gear_num}

def create_record(message):
    return {
        'rear_gear_num':message.get_value('rear_gear_num'),
        'timestamp': message.get_value('timestamp').timestamp(),
        'power': message.get_value('power'),
        'cadence': message.get_value('cadence'),
        'speed': message.get_value('speed'),
        'grade': message.get_value('grade')
    }

def is_gear_change_event(message):
    return message.name == 'event' and message.get_value('event') in ('front_gear_change', 'rear_gear_change')


def calculate_averages(df):
    total_power = df['power'].sum()
    total_cadence = df['cadence'].sum()
    total_speed = df['speed'].sum()
    total_grade = df['grade'].sum()

    count = len(df)
    #write the return of this function that returns the averages make sure to typecast
    return {'power': total_power / count, 'cadence': total_cadence / count, 'speed': total_speed / count, 'grade': total_grade / count}



def average(records, attr):
    values = [r.get(attr, 0) or 0 for r in records]
    return sum(values) / len(values) if values else 0


def process_event(event, records, prevgear):
    relevant_records = pd.DataFrame((filter(lambda r: abs(event['timestamp'] - r['timestamp']) <= 15, records)))

    if not relevant_records.empty:
        avg_values = {
            'power': relevant_records['power'].mean(),
            'cadence': relevant_records['cadence'].mean(),
            'speed': relevant_records['speed'].mean(),
            'grade': relevant_records['grade'].mean(),
            'label': determine_label(event['rear_gear_num'], last_gear_num(prevgear))
        }
        return avg_values
    else:
        return None



def determine_label(rear_gear_num, last_gear):
    if last_gear is None:
        # Handle the first event or no previous gear case
        return 2  # You may choose a different default label if appropriate
    else:
        return 1 if rear_gear_num > last_gear else 0 if rear_gear_num < last_gear else 2

def last_gear_num(records):
    # Finds the rear gear number of the most recent record
    return records[-1]['rear_gear_num']





def create_not_shifting_data_points(records, last_gearchange, time_threshold):
    not_shifting_data_points = []
    for last_event in last_gearchange:
        timestamp = last_event['timestamp']
        relevant_records = pd.DataFrame((filter(lambda r: abs(timestamp - r['timestamp']) <= time_threshold, records)))
        if not relevant_records.empty:
            avg_values = {
                'power': relevant_records['power'].mean(),
                'cadence': relevant_records['cadence'].mean(),
                'speed': relevant_records['speed'].mean(),
                'grade': relevant_records['grade'].mean(),
                'label': 2  # No gear shift
            }
            not_shifting_data_points.append(avg_values)
    return not_shifting_data_points
