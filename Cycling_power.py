'''
Cycling Power is a CircuitPython Library for extracting power from a power meter.
This library can be used to extract the power value in watts
This was created by Ezra Schwartz
It was tested with a Quarq Power Meter and was returning the same values that where found on the wahoo cycling computer that was connected to the power meter at the time of testing.


The power_Values function will return the power value that is sent in the packet

The ByteArray function will return the raw data from the packet.


Example Usage:

        power_sensor = ble.connect(advertisement)

        power_service = power_sensor[CyclingPowerService]

        print(power_service.power_Value)


'''

import struct
from collections import namedtuple
import _bleio
from adafruit_ble.services import Service
from adafruit_ble.uuid import StandardUUID
from adafruit_ble.characteristics import Characteristic, ComplexCharacteristic

try:
    from typing import Optional
except ImportError:
    pass

CPMeasurementValues = namedtuple(
    "CPMeasurementValues",
    (
        "value"
        "ByteArray"
    ),
)


class _CPMeasurement(ComplexCharacteristic):
    """Notify-only characteristic of power data."""
    uuid = StandardUUID(0x2A63)

    def __init__(self) -> None:
        super().__init__(properties=Characteristic.NOTIFY)

    def bind(self, service: "CyclingPowerService") -> _bleio.PacketBuffer:
        """Bind to a CyclingPowerService."""
        bound_characteristic = super().bind(service)
        bound_characteristic.set_cccd(notify=True)
        # Use a PacketBuffer to receive the CPS data
        return _bleio.PacketBuffer(bound_characteristic, buffer_size=1)


class CyclingPowerService(Service):
    """Service for reading from a Cycling Power sensor."""
    uuid = StandardUUID(0x1818)

    cp_measurement = _CPMeasurement()

    def __init__(self, service: Optional["CyclingPowerService"] = None) -> None:
        super().__init__(service=service)
        self._measurement_buf = None

    @property
    def power_Value(self) -> Optional[CPMeasurementValues]:  # returns the power value

        if self._measurement_buf is None:
            self._measurement_buf = bytearray(self.cp_measurement.incoming_packet_length)

        # Clear the buffer
        for i in range(len(self._measurement_buf)):
            self._measurement_buf[i] = 0

        packet_length = self.cp_measurement.readinto(self._measurement_buf)
        if packet_length > 0:

            # find the byte pair for power and return it
            for i in range(0, len(self._measurement_buf), 2):
                if i + 2 <= len(self._measurement_buf):
                    if (i == 2):
                        value = struct.unpack_from('<H', self._measurement_buf, i)[0]
                        return value
                    else:
                        continue

    @property
    def byte_array(self) -> Optional[CPMeasurementValues]:  # returns the raw Byte Array from the packet
        # Clear the buffer

        if self._measurement_buf is None:
            self._measurement_buf = bytearray(self.cp_measurement.incoming_packet_length)

        packet_length = self.cp_measurement.readinto(self._measurement_buf)

        ByteArray = ''.join('{:02x}'.format(x) for x in self._measurement_buf)

        return ByteArray





