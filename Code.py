# Full semi working
# import time
# import json
# from adafruit_ble import BLERadio
# from adafruit_ble.advertising import Advertisement
# from adafruit_ble import UUID  # or from uuid import UUID if using standard library
# from adafruit_ble import BLEConnection
#
# # Load device addresses from the JSON file
# with open(' (CIRCUITPY)Sensors.json', 'r') as file:
#     device_addresses = json.load(file)
# def read_ble_data(device_address, device_name):
#     ble = BLERadio()
#     print(f"Scanning for {device_name} at {device_address}")
#
#     try:
#         for advertisement in ble.start_scan(Advertisement):
#             if advertisement.complete_name == device_address["name"]:  # Compare case-insensitively
#                 print(f"Connecting to {device_name} at {device_address}")
#                 peripheral = ble.connect(advertisement)
#                 print("help(peripheral)")
#                 help(peripheral)
#                 print("print(help(peripheral))")
#
#                 service_uuid = "0x1816"
#                 characteristic_uuid = "0x2A5B"
#                 print("Services")
#                 print(peripheral._discovered_bleio_services["0x1816"])
#                 # Discover the remote service
#                 remote_service = peripheral._discover_remote(service_uuid)
#
#                 # Check if the service was found
#                 if remote_service:
#                     print("Service found:", remote_service)
#                     # You can now interact with the service, read characteristics, etc.
#                 else:
#                     print("Service not found")
#     finally:
#         ble.stop_scan()
#         time.sleep(1)
#
# def main():
#     for device_name, device_address in device_addresses.items():
#         read_ble_data(device_address, device_name)
#
# if __name__ == "__main__":
#     main()
#


from adafruit_ble import BLERadio
from adafruit_ble.advertising import Advertisement
from adafruit_ble.uuid import UUID  # Correct import for UUID
import time
import json

# Load device addresses from the JSON file
with open('Sensors.json', 'r') as file:
    device_addresses = json.load(file)


def read_ble_data(device_address, device_name):
    ble = BLERadio()
    print(f"Scanning for {device_name} at {device_address}")

    try:
        for advertisement in ble.start_scan(Advertisement):
            if advertisement.complete_name == device_address["name"]:
                print(f"Connecting to {device_name} at {device_address}")
                peripheral = ble.connect(advertisement)

                # Use the correct UUID format
                service_uuid = ("00001816-0000-1000-8000-00805f9b34fb")  # Standard 128-bit format for 0x1816

                # Discover the remote service
                remote_service = peripheral._discover_remote(service_uuid)

                # Print discovered services for debugging
                print("Discovered services:", peripheral._discovered_bleio_services.keys())

                # Check if the service was found
                if remote_service:
                    print("Service found:", remote_service)
                else:
                    print("Service not found")

    finally:
        ble.stop_scan()
        time.sleep(1)


def main():
    for device_name, device_address in device_addresses.items():
        read_ble_data(device_address, device_name)


if __name__ == "__main__":
    main()

# error reading Buffer size. no issue with uuid to knowledge

# import time
# import json
# from adafruit_ble import BLERadio
# from adafruit_ble.advertising import Advertisement
# from adafruit_ble.uuid import StandardUUID

# # Load device addresses from the JSON file
# with open(' (CIRCUITPY)Sensors.json', 'r') as file:
#     device_addresses = json.load(file)

# def read_ble_data(device_address, device_name):
#     ble = BLERadio()
#     print(f"Scanning for {device_name} at {device_address}")

#     try:
#         for advertisement in ble.start_scan(Advertisement):
#             if advertisement.complete_name == device_address["name"]:  # Compare case-insensitively
#                 print(f"Connecting to {device_name} at {device_address}")
#                 peripheral = ble.connect(advertisement)

#                 # UUIDs for Cycling Power Service and Characteristic
#                 cps_uuid = StandardUUID('00001818-0000-1000-8000-00805f9b34fb')
#                 cpm_uuid = StandardUUID('00002A63-0000-1000-8000-00805f9b34fb')

#                 cps = peripheral[cps_uuid]
#                 print(cps)
#                 if cpm_uuid in cps.characteristics:
#                     cpm = cps[cpm_uuid]
#                     data = cpm.read()
#                     print(f"Cycling Power Data from {device_name}: {data}")
#                 else:
#                     print("Cycling Power Measurement characteristic not found.")

#                 peripheral.disconnect()
#     finally:
#         ble.stop_scan()
#         time.sleep(1)

# def main():
#     for device_name, device_address in device_addresses.items():
#         read_ble_data(device_address, device_name)

# if __name__ == "__main__":
#     main()


# #Should Work but doesnt and complains about cs chip
# import time
# import json
# import board
# import busio
# import digitalio
# import adafruit_esp32spi.adafruit_esp32spi_socket as socket
# import adafruit_requests as requests
# from adafruit_esp32spi import adafruit_esp32spi
# from adafruit_ble import BLERadio
# from adafruit_ble.advertising.standard import ProvideServicesAdvertisement
# from adafruit_ble.services.standard.device_info import DeviceInfoService
# from adafruit_ble_cycling_speed_and_cadence import CyclingSpeedAndCadenceService
#
# # Load device addresses from the JSON file
# with open('Sensors.json', 'r') as file:
#     device_addresses = json.load(file)
#
# # Set up the ESP32-SPI co-processor
# esp32_cs = digitalio.DigitalInOut(board.ESP_CS)
# esp32_ready = digitalio.DigitalInOut(board.ESP_BUSY)
# esp32_reset = digitalio.DigitalInOut(board.ESP_RESET)
# spi = busio.SPI(board.SCK, board.MOSI, board.MISO)
# esp = adafruit_esp32spi.ESP_SPIcontrol(spi, esp32_cs, esp32_ready, esp32_reset)
# requests.set_socket(socket, esp)
#
# def connect_to_cadence_sensor(device_address, device_name):
#     ble = BLERadio(esp)
#     print(f"Scanning for {device_name} at {device_address}")
#
#     # Define a callback function that will handle the cadence data
#     def cadence_data_handler(sender, data):
#         # The first byte of data is the flags, the second and third bytes are the cumulative crank revolutions,
#         # and the fourth and fifth bytes are the last crank event time
#         # See: https://www.bluetooth.com/wp-content/uploads/Sitecore-Media-Library/Gatt/Xml/Characteristics/org.bluetooth.characteristic.csc_measurement.xml
#         flags = data[0]
#         cumulative_crank_revolutions = data[1] + data[2] * 256
#         last_crank_event_time = data[3] + data[4] * 256
#         print(f"Cadence: {cumulative_crank_revolutions}, {last_crank_event_time}")
#
#     # Create a service object for the cycling speed and cadence service
#     cycling_speed_and_cadence_service = CyclingSpeedAndCadenceService()
#
#     # Create an advertisement object that includes the cycling speed and cadence service
#     advertisement = ProvideServicesAdvertisement(cycling_speed_and_cadence_service)
#
#     # Start scanning for devices that provide the cycling speed and cadence service
#     try:
#         for adv in ble.start_scan(ProvideServicesAdvertisement, timeout=60):
#             if cycling_speed_and_cadence_service in adv.services:
#                 print(f"Found a cadence sensor: {adv.complete_name}")
#                 if adv.complete_name == device_address["name"]:
#                     print(f"Connecting to {device_name} at {device_address}")
#                     # Connect to the device and discover its services and characteristics
#                     peripheral = ble.connect(adv)
#                     print(f"Connected to {device_name} at {device_address}")
#                     # Find the cycling speed and cadence measurement characteristic and start notifications
#                     cycling_speed_and_cadence_measurement_characteristic = peripheral[cycling_speed_and_cadence_service][0]
#                     cycling_speed_and_cadence_measurement_characteristic.start_notify(cadence_data_handler)
#                     print(f"Started notifications for {device_name} at {device_address}")
#                     # Wait for 10 seconds to receive data
#                     time.sleep(10)
#                     # Stop notifications and disconnect
#                     cycling_speed_and_cadence_measurement_characteristic.stop_notify()
#                     print(f"Stopped notifications for {device_name} at {device_address}")
#                     peripheral.disconnect()
#                     print(f"Disconnected from {device_name} at {device_address}")
#                     # Exit the loop
#                     break
#     except Exception as e:
#         print(f"Error scanning for {device_name} at {device_address}: {e}")
#     finally:
#         ble.stop_scan()
#         print(f"Stopped scanning for {device_name} at {device_address}")
#
# def main():
#     for device_name, device_address in device_addresses.items():
#         connect_to_cadence_sensor(device_address, device_name)
#
# if __name__ == "__main__":
#     main()


# Test 1

# ble = BLERadio()
# print(f"Scanning for {device_name} at {device_address}")

# try:
#     print(Advertisement)
#     for advertisement in ble.start_scan(Advertisement, timeout=5):

#         if advertisement.complete_name == device_address["name"]:

#             print(f"Connecting to {device_name} at {device_address}")
#             peripheral = ble.connect(advertisement)


#             print(advertisement.services)
#             print()
#             # Access the remote services offered by the peripheral


#             for service in peripheral.discover_remote_services():
#                 print(f"Service: {service.uuid}")

#                 # Access the characteristics of each service
#                 for characteristic in service.characteristics:
#                     print(f"Characteristic: {characteristic.uuid}")

#                     # Read data from the characteristic if it is readable
#                     if characteristic.properties.read:
#                         data = characteristic.read()
#                         print(f"Data from {device_name} at {device_address}: {data}")

# except Exception as e:
#     print(f"Error: {e}")
# finally:
#     ble.stop_scan()
#     peripheral.disconnect()
#     time.sleep(1)


# Test 2

# ble = BLERadio()
# print(f"Scanning for {device_name} at {device_address}")

# try:
#     for advertisement in ble.start_scan(Advertisement, timeout=5):
#         if advertisement.complete_name == device_address["name"]:
#             print(f"Connecting to {device_name} at {device_address}")
#             peripheral = ble.connect(advertisement)

#             # Discover services offered by the peripheral
#             services = peripheral.discover_services()
#             for service in services:
#                 print(f"Service: {service.uuid}")

#                 # Discover characteristics of each service
#                 characteristics = service.characteristics
#                 for characteristic in characteristics:
#                     print(f"Characteristic: {characteristic.uuid}")

#                     # Read data from the characteristic
#                     if characteristic.properties.read:
#                         data = characteristic.read()
#                         print(f"Data from {device_name} at {device_address}: {data}")

# except Exception as e:
#     print(f"Error: {e}")
# finally:
#     ble.stop_scan()
#     time.sleep(1)


# Test 3


# def read_ble_data(device_address, device_name):
#     ble = BLERadio()
#     print(f"Scanning for {device_name} at {device_address}")

#     try:
#         for advertisement in ble.start_scan(Advertisement):
#             if advertisement.complete_name == device_address["name"]:  # Compare case-insensitively
#                 print(f"Connecting to {device_name} at {device_address}")
#                 peripheral = ble.connect(advertisement)


#                 print(peripheral.services())
#                 for service in peripheral:
#                         for characteristic in service:
#                             print(f"Characteristic: {characteristic}")

#                             # Check if the characteristic has the UUID attribute
#                             if hasattr(characteristic, 'uuid'):
#                                 data = characteristic.read()
#                                 print(f"Data from {device_name} at {device_address}: {data}")
#                             else:
#                                 # If the characteristic doesn't have a UUID attribute, print the raw value
#                                 data = characteristic.read()
#                                 print(f"Data from {device_name} at {device_address}: {data}")

#     finally:
#         ble.stop_scan()
#         time.sleep(1)