import struct

def print_16bit_ints(packet):
    byte_data = bytes.fromhex(packet)

    # Iterate through the packet in 2-byte increments
    for i in range(0, len(byte_data), 2):
        if i + 2 <= len(byte_data):
            # Unpack 2 bytes (16 bits) at the current position
            value = struct.unpack_from('<H', byte_data, i)[0]
            print(f"Value at bytes {i}-{i+1}: {value}")


# Example usage with one of your packets
packet = "2C08 0000 8500 1000 E582 FBDE"

power = print_16bit_ints(packet)
print("Instantaneous Power:", power)
