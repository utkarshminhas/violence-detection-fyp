CC = python

TARGET = main

$(TARGET): $(TARGET).py
	env\Scripts\activate.bat
	$(CC) $(TARGET).py
