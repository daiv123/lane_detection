import serial
import random
if __name__ == '__main__':
    ser = serial.Serial('/dev/cu.usbmodem14301', 9600, timeout=1, write_timeout = 0)
    ser.flush()
    while True:
        servo = input("enter servo: ")
        number = input("enter number: ")
        ser.write((servo+number+'\n').encode('utf-8'))