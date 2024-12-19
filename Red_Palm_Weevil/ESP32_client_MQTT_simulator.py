import paho.mqtt.client as mqtt
import ssl
import subprocess
import time
import os
import random
import json


# Define the MQTT settings
broker = "broker.emqx.io"  # Change as necessary
port = 8883                   # Common port for MQTT
#topic = "test/topic"          # Define your topic
# Specify the topics to subscribe to
topics = [
    ("connect_to_sensors_net/request", 0),   # -->   <--
    ("connect_to_sensors_net/a_place_is_available", 0),   # <--   -->
    ("awake_device", 0),   # <--   -->publish_status
    ("notify_audio_available", 0),   # -->   <--
    ("recording_problems", 0),   # -->   <--
    ("enable_disable_audio_record",0),   # <--   -->
    ("audio/request", 0),   #<--
    ("audio/ack", 0),   #<--
    ("get_infos", 0),   #<--   -->publish_status
    ("sleep_mode", 0),   #<--   -->publish_status
    ("disconnection_from_broker", 0),   #<--   -->publish_status
    ("reset", 0),    #<--   -->publish_status
    ("set_record_frequency", 0)    #<--   -->publish_status
]




class Device:


    def __init__(self, device_model, device_id, location, battery_level, enable_audio_record, recording_frequency, audio_available, last_record_date=None, audio_data_topic_key=None, communication_id=None, invitation_id=None, status=None, ip_address=None):
        """
        Initialize the device with relevant information.

        :param device_model: str - Model of the device
        :param device_id: str - Unique identifier for the device
        :param location: str - Physical location of the device
        :param battery_level: float - Current battery level percentage of the device (0.0 to 100.0)
        :param status: str - Current status of the device ("idle", "waiting_to_record_audio", "waiting_an_audio_request", "sending_audio", "sleeping", "recording_problems")
        :param ip_address: str - IP address of the device in the network
        :param audio_available: boolean - (True/False) it's a flag that describes the availability of an already recorded and validated audio, ready to be sent to the RaspberryPy
        :param audio_data_topic_key: str - random generated key to be appended to "audio/payload" (thus obtaining "audio/payload"+audio_data_topic_key) in order to create a topic on fly and use it to send the audio data towards the RaspberryPy
        :param communication_id: str - last random generated id used in order to be recognizable in a new or an already started communication with the RaspberryPy, for a certain topic
        :param invitation_id: str - random generated id to use in order to be recognizable in an invitation to be part of the sensors network as soon as a place in the net has been provided (the invitation has been sent by the RaspberryPy subsequently to a request to be part of the net made previously by the ESP32). The id will be initially created from the ESP32 for the topic "connect_to_sensors_net/request", and in case reused for the topic "connect_to_sensors_net/place_availabel"
        :param enable_audio_record: boolean - (True/False) it's a flag to enable the audio recording or deactivate it
        :param last_record_date: str - date of the last audio record done
        :param recording_frequency: int - audio recording frequency, that is the amount of time it should elapse between two consecutive records (in seconds)
        # NOT USED :param device_group: int - (1/2/3/4) there are 4 device groups (in this way, a quad-core raspberry could handle 4 devices of 4 different groups together in order to request 4 different audio payload in parallel)
        """
        self.device_model = device_model
        self.device_id = device_id
        self.location = location
        self.battery_level = battery_level
        self.status = status
        self.ip_address = ip_address
        self.audio_available = audio_available
        self.audio_data_topic_key = audio_data_topic_key
        self.communication_id = communication_id
        self.invitation_id = invitation_id
        self.enable_audio_record = enable_audio_record
        self.last_record_date = last_record_date
        self.recording_frequency = recording_frequency
        #self.device_group = device_group

    def update_communication_id(self, communication_id):
        self.communication_id = communication_id

    def update_invitation_id(self, invitation_id):
        self.invitation_id = invitation_id

    def update_audio_data_topic_key(self, audio_data_topic_key):
        self.audio_data_topic_key = audio_data_topic_key

    def update_device_id(self, device_id):
        self.device_id = device_id

    def update_enable_audio_record(self, enable_audio_record):
        self.enable_audio_record = enable_audio_record

    def update_last_record_date(self, last_record_date):
        self.last_record_date = last_record_date

    def update_recording_frequency(self, recording_frequency):
        self.recording_frequency = recording_frequency

    def update_audio_available(self, audio_available):
        self.audio_available = audio_available

    def update_device_group(self, device_group):
        self.device_group = device_group

    def update_battery_level(self, new_level):
        """
        Update the battery level of the device.

        :param new_level: float - New battery level (0.0 to 100.0)
        """
        if 0.0 <= new_level <= 100.0:
            self.battery_level = new_level
        else:
            raise ValueError("Battery level must be between 0.0 and 100.0.")

    def update_status(self, new_status):
        """
        Update the status of the device.

        :param new_status: str - New status of the device
        """
        self.status = new_status

    def all_infos_to_json(self):
        """
        Convert the Client object fields into a JSON string.

        :return: str - JSON string representation of the Client object
        """
        esp32_data = {
            "device_model": self.device_model,
            "device_id": self.device_id,
            "location": self.location,
            "battery_level": self.battery_level,
            "status": self.status,
            "ip_address": self.ip_address,
            "enable_audio_record": self.enable_audio_record,
            "last_record_date": self.last_record_date,
            "recording_frequency": self.recording_frequency
        }
        return json.dumps(esp32_data)

    def subscription_infos_to_json(self):
        """
        Convert the Client object fields into a JSON string.

        :return: str - JSON string representation of the Client object
        """
        esp32_data = {
            "device_model": self.device_model,
            "location": self.location,
            "battery_level": self.battery_level,
            "status": self.status,
            "ip_address": self.ip_address,
            "enable_audio_record": self.enable_audio_record,
            "last_record_date": self.last_record_date,
            "recording_frequency": self.recording_frequency
        }
        return json.dumps(esp32_data)

    def __str__(self):
        return f"Client(Device Model: {self.device_model}, Device ID: {self.device_id}, Location: {self.location}, Battery Level: {self.battery_level}%, Status: {self.status}, IP Address: {self.ip_address}, Audio Available: {self.audio_available}, Audio Data Topic Key: {self.audio_data_topic_key}, Communication ID: {self.communication_id}, Invitation ID: {self.invitation_id}, Enable Audio Record: {self.enable_audio_record}, Last Record Date: {self.last_record_date}, Recording Frequency: {self.recording_frequency})"









# Generate a random integer ID between 1000 and 9999
random_id = random.randint(1000, 9999)
print(random_id)

import random
import string

def generate_random_code(length=20):
    # Create a pool of uppercase, lowercase letters and digits
    characters = string.ascii_letters + string.digits
    # Generate a random code by selecting random characters from the pool
    random_code = ''.join(random.choice(characters) for _ in range(length))
    return random_code

# Generate a 20-digit random alphanumeric code
code = generate_random_code(20)
print("Random Alphanumeric Code:", code)

'''
# Callback function when a message is received.
def on_message(client, userdata, msg):
    print(f"Message received on topic {msg.topic}: {msg.payload.decode('utf-8')}")

    if msg.topic == "connect_to_sensors_net/credentials_assign":
        # Decoding the payload to a string if it's in bytes
        payload = msg.payload.decode('utf-8')

        # Optionally, if you expect the payload to be a valid JSON string or format it:
        try:
            data = json.loads(payload)  # Convert the payload to a Python dictionary
            # Now you can access values using keys
            # For example, if we want to get the value of key "temperature"
            if "temperature" in data:
                temperature = data["temperature"]
                print(f"Temperature: {temperature}")
            else:
                print("Key 'temperature' not found in the JSON data.")
            json_data = json.dumps(data, indent=4)  # Convert it back to a JSON string, nicely formatted
            print(f"Received message in JSON format:\n{json_data}")

        except json.JSONDecodeError:
            print("Received payload is not valid JSON.")
            print(f"Payload: {payload}")
'''










'''

topic: connect_to_sensors_net/request
        ESP32                                                                                       RASPBERRYPY
        _____                                                                                       ______
        
#1      -------------------------------------------------------------------------------------------->{invitation_id:<ID>,infos:{infos}}}

#2      {invitation_id:<ID>, device_id:<ID>, Connection_permission:<allowed/denied>, recording_frequency<freq>}<----------------------------------------------

#3      -------------------------------------------------------------------------------------------->{invitation_id:<ID>,infos:{infos}}}

PS: the infos in #1 are not complete (device_id is not present). Instead infos are re-given in #3, complete with also device_id assigned by the RaspberryPy, so that the RaspberryPy can update the infos of the new device, with key 'device_id'

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

topic: connect_to_sensors_net/a_place_is_available
        ESP32                                                                                       RASPBERRYPY
        _____                                                                                       ______
        
#1      {invitation_id:<ID>, device_id:<ID>, recording_frequency<freq>}<----------------------------------------------

#1      -------------------------------------------------------------------------------------------->{invitation_id:<ID>,infos:{infos}}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

topic: awake_device

        ESP32                                                                                       RASPBERRYPY
        _____                                                                                       ______

#1      {device_id:<ID>, communication_id:<ID>}<--------------------------------------------------
        
#2      -------------------------------------------------------------------------------------------->{communication_id:<ID>,infos:{infos}}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

topic: notify_audio_available

        ESP32                                                                                       RASPBERRYPY
        _____                                                                                       ______

#1      -------------------------------------------------------------------------------------------->{communication_id:<ID>,infos:{infos}}

#2      {communication_id:<ID>}<--------------------------------------------------

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

topic: recording_problems

        ESP32                                                                                       RASPBERRYPY
        _____                                                                                       ______

#1      -------------------------------------------------------------------------------------------->{communication_id:<ID>,infos:{infos}}

#2      {communication_id:<ID>}<--------------------------------------------------

#3      after this, the device will go in state "RECORDING_PROBLEMS", where the only enabled topics will be sleep_mode/reset/disconnection_from_broker/get_status

#4      after that, the RaspberryPy could optionally use one topic between: sleep_mode/reset/disconnection_from_broker/get_status

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

topic: enable_disable_audio_record

        ESP32                                                                                       RASPBERRYPY
        _____                                                                                       ______

#1      {device_id:<ID>, communication_id:<ID>}<--------------------------------------------------
        
#2      -------------------------------------------------------------------------------------------->{communication_id:<ID>,infos:{infos}}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

topic: audio/request

        ESP32                                                                                       RASPBERRYPY
        _____                                                                                       ______

#1      {device_id:<ID>, audio_data_topic_key:<topic_key>, communication_id:<ID>}<--------------------------------------------------


Dynamic topic: audio/payload<key>

        ESP32                                                                                       RASPBERRYPY
        _____                                                                                       ______

#1      -------------------------------------------------------------------------------------------->audio_content

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

topic: audio/ack

        ESP32                                                                                       RASPBERRYPY
        _____                                                                                       ______
        
#1      {communication_id:<ID>, audio_data_topic_key:<topic_key>, ack:ok/resend}<-----------------------------------

Dynamic topic: audio/payload<key>

#2      if ack == resend (it will be used the new dynamic_topic_key for resending the audio)----->{audio_content}

. . . . . . . . . . . . . . . . . . . . .

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

topic: get_infos

        ESP32                                                                                       RASPBERRYPY
        _____                                                                                       ______

#1      {device_id:<ID>, communication_id:<ID>}<--------------------------------------------------
        
#2      -------------------------------------------------------------------------------------------->{communication_id:<ID>,infos:{infos}}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

topic: sleep_mode

        ESP32                                                                                       RASPBERRYPY
        _____                                                                                       ______

#1      {device_id:<ID>, communication_id:<ID>}<--------------------------------------------------
        
#2      -------------------------------------------------------------------------------------------->{communication_id:<ID>,infos:{infos}}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

topic: disconnection_from_broker

        ESP32                                                                                       RASPBERRYPY
        _____                                                                                       ______

#1      {device_id:<ID>, communication_id:<ID>}<--------------------------------------------------
        
#2      -------------------------------------------------------------------------------------------->{communication_id:<ID>,infos:{infos}}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

topic: reset

        ESP32                                                                                       RASPBERRYPY
        _____                                                                                       ______

#1      {device_id:<ID>, communication_id:<ID>}<--------------------------------------------------
        
#2      -------------------------------------------------------------------------------------------->{communication_id:<ID>,infos:{infos}}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

topic: set_recording_frequency

        ESP32                                                                                       RASPBERRYPY
        _____                                                                                       ______

#1      {device_id:<ID>, communication_id:<ID>, recording_frequency:<freq>}<--------------------------------------------------
        
#2      -------------------------------------------------------------------------------------------->{communication_id:<ID>,infos:{infos}}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

topics = [
    ("connect_to_sensors_net/request", 0),   # -->   <--
    ("connect_to_sensors_net/a_place_is_available", 0),   # -->   <--
    ("awake_device", 0),   # <--   -->publish_status
    ("notify_audio_available", 0),   # -->   <--
    ("recording_problems", 0),   # -->   <--
    ("enable_disable_audio_record",0),   # <--   -->
    ("audio/request", 0),   #<--
    ("audio/ack", 0),   #<--
    ("get_infos", 0),   #<--   -->publish_status
    ("sleep_mode", 0),   #<--   -->publish_status
    ("disconnection_from_broker", 0),   #<--   -->publish_status
    ("reset", 0),
    ("set_record_frequency", 0)
]
'''


def reset_device():
    print("Resetting device") #TODO

def publish_infos_with_communication_id(topic, Device, communication_id):
    infos_content = Device.all_infos_to_json()
    infos = {"infos":infos_content, "communication_id": communication_id}
    infos = json.dumps(infos)
    print(f"Sending: {infos}")
    client.publish(topic, infos)

def publish_infos_with_invitation_id(topic, Device, invitation_id):
    #infos_content = Device.subscription_infos_to_json
    infos_content = Device.all_infos_to_json()
    infos = {"infos":infos_content, "invitation_id": invitation_id}
    infos = json.dumps(infos)
    print(f"Sending: {infos}")
    client.publish(topic, infos)

def publish_wav_file(topic, file_path):
    with open(file_path, "rb") as audio_file:
        audio_data = audio_file.read()
        client.publish(topic, audio_data)
        print(f"Published {os.path.basename(file_path)} to topic '{topic}'")

def notify_available_audio_with_communication_id(topic, Device, communication_id):
    publish_infos_with_communication_id(topic, Device, communication_id)

def record_audio_and_notify(Device):
    valid_audio = False # flag True/False to assert the audio is valid/invalid
    max_recording_attempts = 3
    iter_attempts = 1
    while iter_attempts <= max_recording_attempts and valid_audio is False:
        # Start recording
        # Audio validation (check audio correctness) --> if validation is successfully (True) then continue, otherwise (False) retry other 3 times and in the worst case notify the problem
        ###############################################################################
        #The audio recorded can be correct as expected (True) or not (False)
        valid_audio = False # change it to the preferred value for the simulation test (valid_audio = False to simulate a recording problem)
        print("valid_audio: ", valid_audio)
        ###############################################################################
        iter_attempts += 1

    communication_id = generate_random_code(20)
    if valid_audio is False:
        notify_available_audio_with_communication_id("recording_problems", Device, communication_id)
    else:
        notify_available_audio_with_communication_id("notify_audio_available", Device, communication_id)


# Callback function when the client receives a CONNack response from the server.
def on_connect(client, userdata, flags, rc):
    print(f"Connected to broker at {broker} with result code {rc}")
    for topic, qos in topics:
        client.subscribe(topic)
        print(f"Subscribed to topic: {topic}")

'''
topics = [
    ("connect_to_sensors_net/request", 0),   # -->   <--
    ("connect_to_sensors_net/a_place_is_available", 0),   # -->   <--
    ("awake_device", 0),   # <--   -->publish_status
    ("notify_audio_available", 0),   # -->   <--
    ("recording_problems", 0),   # -->   <--
    ("enable_disable_audio_record",0),   # <--   -->
    ("audio/request", 0),   #<--
    ("audio/ack", 0),   #<--
    ("get_infos", 0),   #<--   -->publish_status
    ("sleep_mode", 0),   #<--   -->publish_status
    ("disconnection_from_broker", 0),   #<--   -->publish_status
    ("reset", 0),
    ("set_record_frequency", 0)
]
'''


# Callback function when a message is received.
def on_message(client, userdata, msg, esp32_device):
    print(f"Message received on topic {msg.topic}: {msg.payload.decode('utf-8')}")
    if msg.topic == "connect_to_sensors_net/request":
        # 2 {invitation_id:<ID>, device_id:<ID>, Connection_permission:<allowed/denied>, recording_frequency<freq>}<----------------------------------------------
        json_received = msg.payload.decode('utf-8')
        json_received = json.loads(json_received)  # Convert the payload to a Python dictionary
        try:
            if json_received["invitation_id"] == esp32_device.invitation_id:
                if json_received["Connection_permission"] == "allowed":
                    esp32_device.update_device_id(json_received["device_id"]) # Assignment of the Device ID to the device in the sensors network
                    esp32_device.update_recording_frequency(json_received["recording_frequency"]) # Setting the recording frequency (in seconds) to do audio-recording
                    #esp32_device.update_device_group(json_received["device_group"]) #Assigning a group between 1/2/3/4, which the device from now on will belong to (the group can change for next device_group assignments done through the "set_device_group" topic)
                    publish_infos_with_invitation_id("connect_to_sensors_net/request", esp32_device, json_received["invitation_id"]) # the Raspberry Py will update the infos of this accepted new device
                elif json_received["Connection_permission"] == "denied":
                    pass # waiting for an invitation by the RaspberryPy itself, at the topic: "connect_to_sensors_net/a_place_is_available", using the same Device.communication_id
        except json.JSONDecodeError:
            print("Received payload is not valid JSON.")
            print("ESP32: problem receiving the package for the topic 'connect_to_sensors_net/request'")

    if msg.topic == "connect_to_sensors_net/a_place_is_available":
        # waiting for an invitation by the RaspberryPy itself, at the topic: "connect_to_sensors_net/a_place_is_available", using the same Device.communication_id made up for the previous topic
        # 2 {invitation_id:<ID>, device_id:<ID>, recording_frequency<freq>, device_group:<group>}<----------------------------------------------
        json_received = msg.payload.decode('utf-8')
        json_received = json.loads(json_received)  # Convert the payload to a Python dictionary
        try:
            if json_received["invitation_id"] == esp32_device.invitation_id:
                esp32_device.update_device_id(json_received["device_id"]) # Assignment of the Device ID to the device in the sensors network
                esp32_device.update_recording_frequency(json_received["recording_frequency"])
                #esp32_device.update_device_group(json_received["device_group"])
                publish_infos_with_invitation_id("connect_to_sensors_net/a_place_is_available", esp32_device, json_received["invitation_id"])
        except json.JSONDecodeError:
            print("Received payload is not valid JSON.")
            print("ESP32: problem receiving the package for the topic 'connect_to_sensors_net/a_place_is_available'")

    elif msg.topic == "awake_device":
        # 1      {device_id:<ID>, communication_id:<ID>}<--------------------------------------------------
        json_received = msg.payload.decode('utf-8')
        json_received = json.loads(json_received)  # Convert the payload to a Python dictionary
        try:
            if json_received["device_id"] == esp32_device.device_id:
                esp32_device.update_communication_id(json_received["communication_id"])
                publish_infos_with_communication_id("awake_device", esp32_device, json_received["communication_id"])
                esp32_device.update_enable_audio_record(True)
                '''
                time.sleep(2)
                topics_to_resuscribe = [
                    ("connect_to_sensors_net/request", 0),  # -->   <--
                    ("audio/request", 0),  # <--
                    ("audio/ack", 0),  # <--
                    ("sleep_mode", 0),  # <--   -->publish_status
                ]
                for topic, qos in topics_to_resuscribe:
                    client.subscribe(topic)
                    print(f"Resubscribed to topic: {topic}")
                '''
        except json.JSONDecodeError:
            print("Received payload is not valid JSON.")
            print("ESP32: problem receiving the package for the topic 'awake_device'")

    elif msg.topic == "notify_audio_available":
        # 1      -------------------------------------------------------------------------------------------->{communication_id:<ID>,infos:{infos}}
        # 2      {communication_id:<ID>}<--------------------------------------------------
        json_received = msg.payload.decode('utf-8')
        json_received = json.loads(json_received)  # Convert the payload to a Python dictionary
        try:
            if json_received["communication_id"] == esp32_device.communication_id:
                # disable the audio recording:
                esp32_device.enable_audio_record = False
                '''
                # unsubscribe to every topic but: get_status/disconnection_from_broker/reset/sleep_mode/enable_disable_audio_record
                topics_to_unsuscribe = [
                    ("connect_to_sensors_net/request", 0),  # -->   <--
                    ("connect_to_sensors_net/a_place_is_available", 0),  # -->   <--
                    ("awake_device", 0),  # <--   -->publish_status
                    ("notify_audio_available", 0),  # -->   <--
                    ("recording_problems", 0),  # -->   <--
                    ("audio/request", 0),  # <--
                    ("audio/ack", 0),  # <--
                    ("set_record_frequency", 0)
                ]
                for topic, qos in topics_to_unsuscribe:
                    client.unsubscribe(topic)
                    print(f"Unsubscribed from topic: {topic}")
                '''
        except json.JSONDecodeError:
            print("Received payload is not valid JSON.")
            print("ESP32: problem receiving the package for the topic 'notify_audio_available'")

    elif msg.topic == "recording_problems":
        # 1      ------------------------------------>{communication_id:<ID>,infos:{infos}}
        # 2      {communication_id:<ID>}<--------------------------------------------------
        # 3      after this, the device will go in state "RECORDING_PROBLEMS", where the only enabled topics will be sleep_mode/reset/disconnection_from_broker/get_status
        # 4      after that, the RaspberryPy could optionally use one topic between: sleep_mode/reset/disconnection_from_broker/get_status
        json_received = msg.payload.decode('utf-8')
        json_received = json.loads(json_received)  # Convert the payload to a Python dictionary
        try:
            if json_received["communication_id"] == esp32_device.communication_id:
                esp32_device.update_enable_audio_record(False)
        except json.JSONDecodeError:
            print("Received payload is not valid JSON.")
            print("ESP32: problem receiving the package for the topic 'recording_problems'")

    elif msg.topic == "enable_disable_audio_record":
        # 1      {device_id:<ID>, communication_id:<ID>}<--------------------------------------------------
        # 2      ---------------------------------------------------->{communication_id:<ID>,infos:{infos}}
        json_received = msg.payload.decode('utf-8')
        json_received = json.loads(json_received)  # Convert the payload to a Python dictionary
        try:
            if json_received["communication_id"] == esp32_device.communication_id:
                esp32_device.update_enable_audio_record(False)
        except json.JSONDecodeError:
            print("Received payload is not valid JSON.")
            print("ESP32: problem receiving the package for the topic 'enable_disable_audio_record'")

    elif msg.topic == "audio/request":
        # 1 {device_id:<ID>, dynamic_topic_key:<topic_key>, communication_id:<ID>}<--------------------------------------------------
        json_received = msg.payload.decode('utf-8')
        json_received = json.loads(json_received)  # Convert the payload to a Python dictionary
        try:
            if json_received["device_id"] == esp32_device.device_id:
                esp32_device.update_communication_id(json_received["communication_id"])
                esp32_device.update_audio_data_topic_key(json_received["audio_data_topic_key"])
                # 2      -------------------------------------------------------------------------------------------->{audio_content}
                file_path = "path/to/your/audio_file.wav"  # Change to your .wav file path
                dynamic_topic = "audio/payload" + json_received["audio_data_topic_key"]
                client.subscribe(dynamic_topic)
                publish_wav_file(dynamic_topic, file_path)
        except json.JSONDecodeError:
            print("Received payload is not valid JSON.")
            print("ESP32: problem receiving the package for the topic 'audio/request'")

    elif msg.topic == "audio/ack":
        # 1  {communication_id:<ID>, dynamic_topic_key:<topic_key>, ack:ok/resend}<-----------------------------------
        json_received = msg.payload.decode('utf-8')
        json_received = json.loads(json_received)  # Convert the payload to a Python dictionary
        try:
            if json_received["communication_id"] == esp32_device.communication_id:
                if json_received["ack"] == "ok":
                    topic_to_unsuscribe = "audio/payload" + esp32_device.audio_data_topic_key
                    client.unsubscribe(topic_to_unsuscribe)
                elif json_received["ack"] == "resend":
                    esp32_device.update_audio_data_topic_key(json_received["audio_data_topic_key"])
                    # Dynamic topic: audio/payload<audio_data_topic_key>
                    # 2      if ack == resend (it will be used the new dynamic_topic_key for resending the audio)----->{audio_content}
                    file_path = "path/to/your/audio_file.wav"  # Change to your .wav file path
                    topic_to_unsuscribe = "audio/payload" + esp32_device.audio_data_topic_key
                    client.unsubscribe(topic_to_unsuscribe)
                    dynamic_topic = "audio/payload" + json_received["audio_data_topic_key"]
                    client.subscribe(dynamic_topic)
                    publish_wav_file(dynamic_topic, file_path)
        except json.JSONDecodeError:
            print("Received payload is not valid JSON.")
            print("ESP32: problem receiving the package for the topic 'audio/ack'")

    # elif msg.topic == "audio/payload" + topic_key:                # only for the RaspberryPy

    elif msg.topic == "get_infos":
        # 1      {device_id:<ID>, communication_id:<ID>}<--------------------------------------------------
        json_received = msg.payload.decode('utf-8')
        json_received = json.loads(json_received)  # Convert the payload to a Python dictionary
        try:
            if json_received["device_id"] == esp32_device.device_id:
                esp32_device.update_communication_id(json_received["communication_id"])
                publish_infos_with_communication_id("get_infos", esp32_device, json_received["communication_id"])
        except json.JSONDecodeError:
            print("Received payload is not valid JSON.")
            print("ESP32: problem receiving the package for the topic 'get_infos'")

    elif msg.topic == "sleep_mode":
        # 1      {device_id:<ID>, communication_id:<ID>}<--------------------------------------------------
        json_received = msg.payload.decode('utf-8')
        json_received = json.loads(json_received)  # Convert the payload to a Python dictionary
        try:
            if json_received["device_id"] == esp32_device.device_id:
                esp32_device.update_communication_id(json_received["communication_id"])
                publish_infos_with_communication_id("sleep_mode", esp32_device, json_received["communication_id"])
                esp32_device.update_enable_audio_record(False)
                '''
                topics_to_unsuscribe = [
                    ("connect_to_sensors_net/request", 0),  # -->   <--
                    ("audio/request", 0),  # <--
                    ("audio/ack", 0),  # <--
                    ("sleep_mode", 0),  # <--   -->publish_status
                ]
                for topic, qos in topics_to_unsuscribe:
                    client.unsubscribe(topic)
                    print(f"Unsubscribed from topic: {topic}")
                '''
        except json.JSONDecodeError:
            print("Received payload is not valid JSON.")
            print("ESP32: problem receiving the package for the topic 'sleep_mode'")

    elif msg.topic == "disconnection_from_broker":
        # 1      {device_id:<ID>, communication_id:<ID>}<--------------------------------------------------
        json_received = msg.payload.decode('utf-8')
        json_received = json.loads(json_received)  # Convert the payload to a Python dictionary
        try:
            if json_received["device_id"] == esp32_device.device_id:
                esp32_device.update_communication_id(json_received["communication_id"])
                publish_infos_with_communication_id("disconnection_from_broker", esp32_device, json_received["communication_id"])
                client.disconnect() # Disconnect from the mqtt broker
                esp32_device.update_enable_audio_record(False)
        except json.JSONDecodeError:
            print("Received payload is not valid JSON.")
            print("ESP32: problem receiving the package for the topic 'disconnection_from_broker'")

    elif msg.topic == "reset":
        # 1      {device_id:<ID>, communication_id:<ID>}<--------------------------------------------------
        json_received = msg.payload.decode('utf-8')
        json_received = json.loads(json_received)  # Convert the payload to a Python dictionary
        try:
            if json_received["device_id"] == esp32_device.device_id:
                esp32_device.update_communication_id(json_received["communication_id"])
                publish_infos_with_communication_id("reset", esp32_device, json_received["communication_id"])
                reset_device() # Resetting the device, so everything will be restarted from the beginning
        except json.JSONDecodeError:
            print("Received payload is not valid JSON.")
            print("ESP32: problem receiving the package for the topic 'reset'")

    elif msg.topic == "set_recording_frequency":
        # 1      {device_id:<ID>, communication_id:<ID>, recording_frequency:<freq>}<--------------------------------------------------
        json_received = msg.payload.decode('utf-8')
        json_received = json.loads(json_received)  # Convert the payload to a Python dictionary
        try:
            if json_received["device_id"] == esp32_device.device_id:
                esp32_device.update_recording_frequency(json_received["recording_frequency"])
                esp32_device.update_communication_id(json_received["communication_id"])
                publish_infos_with_communication_id("set_recording_frequency", esp32_device, json_received["communication_id"])
        except json.JSONDecodeError:
            print("Received payload is not valid JSON.")
            print("ESP32: problem receiving the package for the topic 'set_recording_frequency'")

    '''
    elif msg.topic == "set_device_group":
        # 1      {device_id:<ID>, communication_id:<ID>, device_group:<group>}<--------------------------------------------------
        json_received = msg.payload.decode('utf-8')
        json_received = json.loads(json_received)  # Convert the payload to a Python dictionary
        try:
            if json_received["device_id"] == esp32_device.device_id:
                esp32_device.update_recording_frequency(json_received["recording_frequency"])
                esp32_device.update_communication_id(json_received["communication_id"])
                esp32_device.update_device_group(json_received["device_group"])
                publish_infos_with_communication_id("set_device_group", esp32_device, json_received["communication_id"])
        except json.JSONDecodeError:
            print("Received payload is not valid JSON.")
            print("ESP32: problem receiving the package for the topic 'set_device_group'")
    '''

# Create an MQTT client instance
client = mqtt.Client()

# Set the username and password
client.username_pw_set("your_username", "your_password")

# Set SSL parameters
client.tls_set(ca_certs="path/to/ca.crt",  # CA file
               certfile="path/to/client.crt",  # Client certificate
               keyfile="path/to/client.key",  # Client private key
               tls_version=ssl.PROTOCOL_TLS)

# Attach the callback functions
client.on_connect = on_connect
client.on_message = on_message

# Connect to the MQTT broker (using port 8883 for SSL)
client.connect("your_broker_host", 8883)
#client.connect(broker, port, 60)

# Start the loop in a separate thread to process network traffic and dispatch callbacks
client.loop_start()
time.sleep(20)

# Start asking to be part of the sensors network
publish_infos_with_invitation_id("connect_to_sensors_net/request", Device, generate_random_code(20))


# Keep the script running to listen for incoming messages
try:
    while True:
        pass  # You can replace this with other logic if necessary
except KeyboardInterrupt:
    print("Exiting...")
finally:
    client.loop_stop()  # Stop the loop gracefully
    client.disconnect()  # Disconnect from the broker







if __name__ == "__main__":
    esp32_device = Device(
        device_model="ESP32_model_X",
        device_id=None, # must be assigned by the RaspberryPy after an allowed acceptance of the device in the sensors network
        location=None, # the inizialization value for 'location' must be read from an inizialization topic that gives this info
        battery_level=75.5, # use an appropriate function to get the battery level
        status="idle", # the default state is 'idle' (state assigned before being accepted into the sensors network)
        ip_address="192.168.1.100", # use an appropriate function to get the ip address
        audio_available=False,
        audio_data_topic_key=None,
        communication_id=None,
        invitation_id=None,
        enable_audio_record=False,
        last_record_date=None,
        recording_frequency=300 # default value is one audio record every five minutes (300 sec)
    )


    print(esp32_device)

    # Update battery level and status
    # esp32_device.update_battery_level(65.0)
    # esp32_device.update_status("idle")
