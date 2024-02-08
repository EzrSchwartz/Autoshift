import serial

# Define the serial port and baud rate
serial_port = 'COM21'  # Replace 'COM3' with your actual serial port name
baud_rate = 115200  # Use the baud rate used in your CircuitPython code

try:
    # Open the serial port
    ser = serial.Serial(serial_port, baud_rate, timeout=1)
    print(f"Connected to {serial_port} at {baud_rate} baud")

    # Read and display serial data
    while True:
        data = ser.readline().decode('utf-8').strip()
        if data:
            print(data)

except serial.SerialException as e:
    print(f"Serial port error: {e}")
except KeyboardInterrupt:
    pass
finally:
    try:
        # Close the serial port
        ser.close()
    except NameError:
        pass
