import datetime
import os

current_directory = os.getcwd()
current_time = datetime.datetime.now()
print(f"Hello, World! How're you. today is: {current_time}. You are working under {current_directory}" )

try:
    num = int(input("input number: "))
    print("You input: ", num)
except:
    print("Invalid input")