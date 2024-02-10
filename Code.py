from adafruit_ble import BLERadio
from adafruit_ble.advertising import Advertisement
from adafruit_ble_cycling_speed_and_cadence import CyclingSpeedAndCadenceService
import time
import json
import gc
import _bleio 
import board
from cyclingPower import CyclingPowerService, _CPMeasurement
from digitalio import DigitalInOut, Direction, Pull

prev_Tot_Time = 0
prev_Tot_Power = 0
Tot_power = 0
Tot_time = 0
prevtime = 0
current_time = 0
# Load device addresses from the JSON file
with open('Sensors.json', 'r') as file:
    device_addresses = json.load(file)
with open(' (CIRCUITPY)Weights.json', 'r') as file:
    model_data = json.load(file)
shiftup_input_pin = DigitalInOut(board.A4)
shiftup_input_pin.direction = Direction.INPUT
shiftup_input_pin.pull = Pull.UP

# Define the output pin
shiftup_output_pin = DigitalInOut(board.A5)
shiftup_output_pin.direction = Direction.OUTPUT

shiftdown_input_pin = DigitalInOut(board.A0)
shiftdown_input_pin.direction = Direction.INPUT
shiftdown_input_pin.pull = Pull.UP

# Define the output pin
shiftdown_output_pin = DigitalInOut(board.A1)
shiftdown_output_pin.direction = Direction.OUTPUT


def dot_product(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))


def add_vectors(v1, v2):
    return [x + y for x, y in zip(v1, v2)]


def predict(features, model_data):
    # Check for None in features
    if any(f is None for f in features):
        raise ValueError("Input features cannot contain None values.")

    # Extract scaler parameters and check for None
    mean = model_data.get('scaler_mean')
    std = model_data.get('scaler_scale')
    if mean is None or std is None:
        raise ValueError("Scaler parameters (mean, scale) are missing in model_data.")

    # Standardize features
    standardized_features = [(f - m) / s for f, m, s in zip(features, mean, std)]

    # Forward pass through the model
    # Correct forward pass computation for the first layer as an example
    z1 = [relu(sum(f * w for f, w in zip(standardized_features, weights)) + b) for weights, b in
          zip(model_data['weights_fc1'], model_data['biases_fc1'])]
    z2 = [relu(sum(f * w for f, w in zip(standardized_features, weights)) + b) for weights, b in
          zip(model_data['weights_fc2'], model_data['biases_fc1'])]
    logits = [sum(f * w for f, w in zip(z2, weights)) + b for weights, b in zip(model_data['weights_fc3'], model_data['biases_fc3'])]

    # Apply softmax to logits
    probabilities = softmax(logits)

    # Predict label
    predicted_label = probabilities.index(max(probabilities))
    return predicted_label, probabilities


def relu(x):
    """ReLU activation function."""
    return max(0, x)


def softmax(logits):
    """Softmax function applied to a list of logits."""
    max_logit = max(logits)
    exps = [exp(l - max_logit) for l in logits]  # Shift for numerical stability
    sum_of_exps = sum(exps)
    return [e / sum_of_exps for e in exps]


def exp(x):
    """A simple exponentiation function."""
    return 2.718281828459045 ** x  # Using Euler's number for exponentiation


def upshift():
    shiftup_output_pin.value = shiftup_input_pin.value
    print("Shifting UP")


def downshift():
    shiftdown_output_pin.value = shiftdown_input_pin.value
    print("Shifting DOWN")


def calculate_speed(cumulative_wheel_revolutions_current, cumulative_wheel_revolutions_previous, time_current, time_previous, wheel_circumference_mm=2800):

    # Calculate the number of revolutions in this interval
    revolutions = cumulative_wheel_revolutions_current - cumulative_wheel_revolutions_previous

    # Calculate the time interval in hours
    time_interval_hours = (time_current - time_previous) / (1000 * 60 * 60)  # converting milliseconds to hours

    # Check for valid time interval
    if time_interval_hours <= 0:
        return 0  # Return 0 to indicate an invalid calculation

    # Calculate distance traveled in this interval in miles
    distance_miles = (revolutions * wheel_circumference_mm / 1000) / 1609.344  # Convert mm to miles

    # Calculate speed (mph)
    speed_mph = distance_miles / time_interval_hours

    return speed_mph

def handle_cycling_power_notification(data):
    # Assuming data is a bytearray containing the notification value
    # Print the raw data or process it as required
    print("Received Data:", data)


def set_prev_vals(hex):
    for i, byte_pair in enumerate(hex):
        decimal_value = int(byte_pair, 16)
        if i == 2:
            prev_Tot_Time == decimal_value
        if i == 3:
            prev_Tot_Power == decimal_value

def set_current_vars(hex):
    for i, byte_pair in enumerate(hex):
        decimal_value = int(byte_pair, 16)
        if i == 2:
            Tot_time == decimal_value
        if i == 3:
            Tot_power == decimal_value


def print_16bit_ints(packet):
    byte_data = bytes.fromhex(packet)

    # Iterate through the packet in 2-byte increments
    for i in range(0, len(byte_data), 2):
        if i + 2 <= len(byte_data):
            # Unpack 2 bytes (16 bits) at the current position
            value = struct.unpack_from('<H', byte_data, i)[0]
            print(f"Value at bytes {i}-{i+1}: {value}")


def read_ble_data_power(device_name):
        ble = BLERadio()
        print(f"Scanning for {device_name}")
        try:
            for advertisement in ble.start_scan(Advertisement, timeout=5):
                if advertisement.complete_name == device_name:
                    print(f"Found {device_name}, trying to connect...")
                    power_sensor = ble.connect(advertisement)
                    print("Connected.")
                    return power_sensor
                    # if CyclingPowerService in power_sensor:
                    #     power_service = power_sensor[CyclingPowerService]
                    #     # Keep the script running as long as the peripheral is connected
                    #     while power_sensor.connected:
                            
                    #         power_service = peripheral[CyclingPowerService]
                    #         print(power_service.power_Value)
                    # else:
                    #     print(f"{device_name} does not have a Cycling")
                    



            time.sleep(1)  # Short delay before trying again
        except Exception as e:
            print(f"An error occurred: {e}")
            power_sensor.disconnect()
        finally:
            ble.stop_scan()  # Ensure the BLE scan is stopped
            




def read_ble_data_cadence_speed(device_address, device_name):
    gc.enable()
    ble = BLERadio()
    print(f"Scanning for {device_name} at {device_address}")
    try:
        for advertisement in ble.start_scan(Advertisement, timeout=5):
            if advertisement.complete_name == device_address["name"]:
                print(f"Connecting to {device_name} at {device_address}")
                cadence_sensor = ble.connect(advertisement)
                print("connected")
                return cadence_sensor
                # help(peripheral)
                
                # if CyclingSpeedAndCadenceService in cadence_sensor:
                #     cadence_service = cadence_sensor[CyclingSpeedAndCadenceService]
                #     while True:
                #         measurement_values = cadence_service.measurement_values
                #         if measurement_values is not None:
                #             print(measurement_values)
                #             cadence = ((measurement_values.cumulative_crank_revolutions-previousrevolutions)/((measurement_values.last_crank_event_time-previoustime)+1)*60*1024)

                #             speed = calculate_speed(measurement_values.cumulative_wheel_revolutions,previouswheelrevolutions,measurement_values.last_wheel_event_time,previousewheeltime)
                #             previousrevolutions = measurement_values.cumulative_crank_revolutions
                #             previoustime = measurement_values.last_crank_event_time
                #             previouswheelrevolutions =measurement_values.cumulative_wheel_revolutions
                #             previousewheeltime = measurement_values.last_wheel_event_time
                #             print("cadence")
                #             print(cadence)
                #             print("Speed")
                #             print(speed)
                #         time.sleep(3)

            time.sleep(1)
    finally:
        ble.stop_scan()
        time.sleep(1)

def read_sensors_and_shift(cadence_sensor,power_sensor):
    previousrevolutions = 0
    previoustime = 0
    previouswheelrevolutions = 0
    previousewheeltime = 0
    current_time = time.time()
    prevtime = 0
    lastshiftingtime = 0
    if CyclingSpeedAndCadenceService in cadence_sensor and CyclingPowerService in power_sensor:
        cadence_service = cadence_sensor[CyclingSpeedAndCadenceService]
        power_service = power_sensor[CyclingPowerService]

        while power_sensor.connected and cadence_sensor.connected:
            measurement_values = cadence_service.measurement_values
            if measurement_values is not None:
                
                
                cadence = ((measurement_values.cumulative_crank_revolutions-previousrevolutions)/((measurement_values.last_crank_event_time-previoustime)+1)*60*1024)

                speed = calculate_speed(measurement_values.cumulative_wheel_revolutions,previouswheelrevolutions,measurement_values.last_wheel_event_time,previousewheeltime)
                previousrevolutions = measurement_values.cumulative_crank_revolutions
                previoustime = measurement_values.last_crank_event_time
                previouswheelrevolutions =measurement_values.cumulative_wheel_revolutions
                previousewheeltime = measurement_values.last_wheel_event_time

                power = power_service.power_Value
            
                if(speed is not None and cadence is not None and power is not None):
                    current_time = time.time()
                    if (current_time - prevtime >= 5):
                        features = [power, cadence, speed, (current_time - lastshiftingtime)]  # Replace with actual values
                        predicted_label, probabilities = predict(features, model_data)
                        if (predicted_label == 0):
                            print("STAY")
                            continue
                        if (predicted_label == 1 & probabilities > 85):
                            upshift()
                            lastshiftingtime = time.time()

                        if (predicted_label == 2 & probabilities > 85):
                            downshift()
                            lastshiftingtime = time.time()
                        prevtime = time.time()


            time.sleep(1)

        else:
            print(f"{device_name} does not have a Cycling")



def main():
    cadSense = None
    powSense = None
    while True:
        for device_name, device_address in device_addresses.items():
            if(device_name == "cadence_sensor") and cadSense == None:
                cadSense = read_ble_data_cadence_speed(device_address, device_name)
            if(device_name == "power_meter") and powSense == None:
                powSense = read_ble_data_power(device_address["name"])
        if powSense != None and cadSense != None:
            read_sensors_and_shift(cadSense,powSense)


while __name__ == "__main__":
    main()

