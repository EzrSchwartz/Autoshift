import tkinter as tk
from tkinter import filedialog, ttk
import json
import psutil
import win32api
import datetime
import asyncio

try:
    import bleak
except ImportError:
    bleak = None

from FitExtract import process_files
from Train import Training_MLGrades
from TrainNoGrade import Training_ML


def select_files():
    selected_files = filedialog.askopenfilenames(
        title="Select FIT Files",
        filetypes=(("FIT Files", "*.fit"), ("All Files", "*.*"))
    )
    print("Selected Files:")
    for file_path in selected_files:
        print(file_path)
    files.extend(selected_files)
    print(f"{len(selected_files)} files selected.")


def start_processing():
    global model_data, scaler
    if files:
        timestamp, powers, cadences, labels,timelastshift, grades,speeds = process_files(files)
        grade = input("Do you want to use elevation to train and sense: Y/N")

        if grade == "Y":
            model_data, scaler = Training_MLGrades(powers, cadences, speeds, grades, labels, timelastshift)
        if grade == "n" or grade == "N":
            print(labels)
            model_data, scaler = Training_ML(powers, cadences, speeds, labels, timelastshift)

        print("model_data", datetime.datetime.now())
    else:
        print("No files selected.")


def upload_to_selected_port(port, model_data):
    try:
        with open(port + 'Weights.json', 'w') as outfile:
            json.dump(model_data, outfile)
        print(f"Model data uploaded to {port}")
    except Exception as e:
        print(f"Error uploading to {port}: {e}")


def upload_to_selected_port_Sensors(port, model_data):
    try:
        with open(port + 'Sensors.json', 'w') as outfile:
            json.dump(model_data, outfile)
        print(f"Model data uploaded to {port}")
    except Exception as e:
        print(f"Error uploading to {port}: {e}")


def get_drives():
    drives = []
    for partition in psutil.disk_partitions():
        if 'removable' in partition.opts or 'fixed' in partition.opts:
            try:
                drive_name = win32api.GetVolumeInformation(partition.device)[0]
                drives.append(f"{partition.device} ({drive_name})")
            except Exception as e:
                print(f"Error accessing drive {partition.device}: {e}")
    return drives


def on_upload_clicked():
    port = selected_port.get()
    if port and model_data is not None:
        upload_to_selected_port(port, model_data)
        print("Upload Success")
    else:
        print("No port selected or model data is empty.")


async def search_ble_devices():
    if bleak is not None:
        discovered_devices = []
        try:
            devices = await bleak.discover()
            for device in devices:
                device_info = {
                    "address": device.address,
                    "name": device.name if device.name else "Unknown"

                }
                discovered_devices.append(device_info)
                print(f"Discovered BLE Device: {device_info}")
        except Exception as e:
            print(f"Error scanning BLE devices: {e}")
    else:
        print("The 'bleak' library is not available. Install it using 'pip install bleak'.")
        # If bleak is not available, simulate the device discovery
        discovered_devices = [
            {"address": "00:11:22:33:44:55", "name": "Device1"},
            {"address": "11:22:33:44:55:66", "name": None},
        ]

    return discovered_devices


def on_search_ble_clicked():
    discovered_devices = asyncio.run(search_ble_devices())

    # Clear previous text and items in the Text widget and Comboboxes
    text_output.delete(1.0, tk.END)

    # Create separate lists to hold values for each Combobox
    power_meter_values = []
    cadence_sensor_values = []
    speed_sensor_values = []
    global deviceNames
    global deviceAddresses
    deviceNames = []
    deviceAddresses = []
    # Display the discovered devices in the Text widget and add them to the Comboboxes
    for device_info in discovered_devices:
        text_output.insert(tk.END, f"Address: {device_info['address']}, Name: {device_info['name']}\n")
        deviceAddresses.append(device_info['address'])
        deviceNames.append(device_info['name'])
        # Add device address to the Power Meter Combobox
        power_meter_values.append(device_info['name'])

        # Add device address to the Cadence Sensor Combobox
        cadence_sensor_values.append(device_info['name'])

        # Add device address to the Speed Sensor Combobox
        speed_sensor_values.append(device_info['name'])

    # Update the Comboboxes with the new values
    power_meter_combobox['values'] = power_meter_values
    cadence_sensor_combobox['values'] = cadence_sensor_values
    speed_sensor_combobox['values'] = speed_sensor_values

    # Clear the selection in the Comboboxes
    power_meter_combobox.current('')
    cadence_sensor_combobox.current('')
    speed_sensor_combobox.current('')

def get_address_from_name(device_name):
    for i in range(len(deviceNames)):
        if deviceNames[i] == device_name:
            return deviceAddresses[i]
    return "Unknown"


# Function to set the chosen power meter
def set_power_meter():
    chosen_power_meter = power_meter_combobox.get()
    print(f"Chosen Power Meter: {chosen_power_meter}")


# Function to set the chosen cadence sensor
def set_cadence_sensor():
    chosen_cadence_sensor = cadence_sensor_combobox.get()
    print(f"Chosen Cadence Sensor: {chosen_cadence_sensor}")


# Function to set the chosen speed sensor
def set_speed_sensor():
    chosen_speed_sensor = speed_sensor_combobox.get()
    print(f"Chosen Speed Sensor: {chosen_speed_sensor}")

def savebluetoothdevices(deviceData):
    port = selectedDrive.get()

    if port and deviceData is not None:
        upload_to_selected_port_Sensors(port, deviceData)
        print("Upload Success")
    else:
        print("No port selected or model data is empty.")

def save_selected_devices():
    chosen_power_meter_name = power_meter_combobox.get()
    chosen_cadence_sensor_name = cadence_sensor_combobox.get()
    chosen_speed_sensor_name = speed_sensor_combobox.get()

    chosen_power_meter_address = get_address_from_name(chosen_power_meter_name)
    chosen_cadence_sensor_address = get_address_from_name(chosen_cadence_sensor_name)
    chosen_speed_sensor_address = get_address_from_name(chosen_speed_sensor_name)

    selected_devices = {
        "power_meter": {"name": chosen_power_meter_name, "address": chosen_power_meter_address},
        "cadence_sensor": {"name": chosen_cadence_sensor_name, "address": chosen_cadence_sensor_address},
        "speed_sensor": {"name": chosen_speed_sensor_name, "address": chosen_speed_sensor_address},
    }
    savebluetoothdevices(selected_devices)

    # file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
    # if file_path:
    #     with open(file_path, 'w') as outfile:
    #         json.dump(selected_devices, outfile)
    #     print(f"Selected devices saved to {file_path}")

# Create the main window
root = tk.Tk()
root.title("Autoshift")

# Define global variables
files = []
model_data = None
scaler = None
device_address_dict = {}  # Dictionary to map device names to their addresses

# Create a notebook for tabs
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

# Existing FIT file processing tab
fit_tab = ttk.Frame(notebook)
notebook.add(fit_tab, text="FIT File Processing")

# FIT file processing buttons
select_button = tk.Button(fit_tab, text="Select FIT Files", command=select_files)
select_button.pack()

process_button = tk.Button(fit_tab, text="Process Files", command=start_processing)
process_button.pack()

selected_port = tk.StringVar()
ports = get_drives()

selectedDrive = tk.StringVar()
drives = get_drives()

# FIT file upload tab
upload_tab = ttk.Frame(notebook)
notebook.add(upload_tab, text="FIT File Upload")

# FIT file upload buttons
port_label = tk.Label(upload_tab, text="Select a Drive/Port:")
port_label.pack()

port_combobox = ttk.Combobox(upload_tab, textvariable=selected_port, values=ports)
port_combobox.pack()

upload_button = tk.Button(upload_tab, text="Upload to Microcontroller", command=on_upload_clicked)
upload_button.pack()

# BLE sensor discovery tab
ble_tab = ttk.Frame(notebook)
notebook.add(ble_tab, text="BLE Sensors")

# BLE sensor search button
search_ble_button = tk.Button(ble_tab, text="Search BLE Sensors", command=on_search_ble_clicked)
search_ble_button.pack()

# Text widget to display discovered devices
text_output = tk.Text(ble_tab, height=10, width=50)
text_output.pack()

# Combobox to select the power meter
power_meter_combobox_label = tk.Label(ble_tab, text="Select Power Meter:")
power_meter_combobox_label.pack()

power_meter_combobox = ttk.Combobox(ble_tab, values=[], state="readonly")
power_meter_combobox.pack()

# Button to set the chosen power meter
set_power_meter_button = tk.Button(ble_tab, text="Set Power Meter", command=set_power_meter)
set_power_meter_button.pack()

# Combobox to select the cadence sensor
cadence_sensor_combobox_label = tk.Label(ble_tab, text="Select Cadence Sensor:")
cadence_sensor_combobox_label.pack()

cadence_sensor_combobox = ttk.Combobox(ble_tab, values=[], state="readonly")
cadence_sensor_combobox.pack()

# Button to set the chosen cadence sensor
set_cadence_sensor_button = tk.Button(ble_tab, text="Set Cadence Sensor", command=set_cadence_sensor)
set_cadence_sensor_button.pack()

# Combobox to select the speed sensor
speed_sensor_combobox_label = tk.Label(ble_tab, text="Select Speed Sensor:")
speed_sensor_combobox_label.pack()

speed_sensor_combobox = ttk.Combobox(ble_tab, values=[], state="readonly")
speed_sensor_combobox.pack()

# Button to set the chosen speed sensor
set_speed_sensor_button = tk.Button(ble_tab, text="Set Speed Sensor", command=set_speed_sensor)
set_speed_sensor_button.pack()


avaliblePorts = ttk.Combobox(ble_tab, textvariable=selectedDrive, values=ports)
avaliblePorts.pack()

select_drive = tk.Button(ble_tab, text="Upload To Autoshift Drive", command=save_selected_devices)
select_drive.pack()

# Run the GUI
root.mainloop()