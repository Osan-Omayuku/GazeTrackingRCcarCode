#import time

#t_end = time.time() + 10
#while time.time() < t_end:
#    print("Running")

import serial
import time
ser = serial.Serial('COM5', 9600, timeout=0.5)


while (True):
    message = input("Direction: ")
    if message == 'e':
        exit()

    ser.write(message.encode('utf-8'))

    time.sleep(1)

    result = ser.readline()

    print(result)




    


    #time.sleep(1)


#ser.close()