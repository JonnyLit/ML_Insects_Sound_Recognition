#!/usr/bin/env python
import time
start_time = time.time()  # Record the start time
import ssl
import random
import string
import json
import numpy as np
seed = 2018
np.random.seed(seed)
import os
import numpy as np
import librosa
import sys
import subprocess # to call C++ executables
import threading
print("Python version:", sys.version)
print("Python version info:", sys.version_info)
print("NumPy version:", np.__version__)

##############################################################
import paho.mqtt.client as mqtt


# Define the MQTT settings
broker = "broker.emqx.io"  # Change as necessary
port = 8883                   # Common port for MQTT
#topic = "test/topic"          # Define your topic
# Specify the topics to subscribe to
topics = [
    ("audio/wav", 0),   # Subscribe to WAV audio messages
    ("audio/mp3", 0),   # Subscribe to MP3 audio messages
    ("test/topic", 0)   # Subscribe to any other topic
]



# Specify the topics to subscribe to
topics = [
    ("connect_to_sensors_net/ack", 0),   # -->   <--
    ("connect_to_sensors_net/place_available", 0),   # -->   <--
    ("awake_device", 0),   # <--   -->publish_status
    ("audio/request", 0),   #<--
    ("audio/ack", 0),   #<--
    ("get_infos", 0),   #<--   -->publish_status
    ("sleep_mode", 0),   #<--   -->publish_status
    ("disconnection_from_broker", 0),   #<--   -->publish_status
    ("reset", 0),
    ("set_record_frequency", 0)
]


class RaspberryPy:


    def __init__(self, device_model, device_id, num_current_devices, num_max_allowed_devices, num_max_pending_subscription_requests, devices_infos_dict , start_time_sending_audio_request, max_waiting_time_for_receiving_audio, max_resending_request, recording_frequency, audio_data_topic_key=None, communication_id=None, invitation_id_list=None, status=None, ip_address=None, devices_with_ready_audio_to_send_list=None, start_index_ready_devices_list=None, threads_communication_id_dict=None, thread_audio_data_topic_key_dict=None, go_on_with_audio_requests_dict=None, counters_resend_request_done_per_thread_dict=None, current_device_id_handled_by_thread_dict=None):
        """
        Initialize the Raspberry Py with relevant information.

        :param device_model: str - Model of the Raspberry Py
        :param device_id: str - Unique identifier for the Raspberry Py
        :param num_current_devices: int - number of current devices making part of the sensors network
        :param num_max_allowed_devices: int - max number of allowed devices in the sensors network
        :param num_max_pending_subscription_requests: int - max number pending requests of devices that ask for being part of the sensors network (max dimension allowed for 'invitation_id_list')
        :param status: str - Current status of the Raspberry Py (e.g., "idle", "active", "sleeping", etc.)
        :param ip_address: str - IP address of the Raspberry Py in the network
        :param max_waiting_time_for_receiving_audio: int - max amount of time in seconds to wait for an audio to be sent from the ESP32 towards the RaspberryPy
        :param start_time_sending_audio_request: int - time in seconds epoch unit when the request for audio data has been sent from the RaspberryPy to the ESP32
        :param audio_data_topic_key: str - random generated key to be appended to "audio/payload" (thus obtaining "audio/payload"+audio_data_topic_key) in order to create a topic on fly and use it by the ESP32 to send the audio data towards the RaspberryPy
        :param communication_id: str - random generated id to use in order to be recognizable in a new or an already started communication with the ESP32, for a certain topic
        :param recording_frequency: int - audio recording frequency for the audio recording devices, that is the amount of time it should elapse between two consecutive records (in seconds)
        :param devices_infos_dict: dict - dictionary containing the infos of all the net's devices in json format, like: {"device_id_1":{"device_model":<dev_model>, "location":<location>, ...}, ..., "device_id_n":{...}}
        :param invitation_id_list: list - queue list of the several invitation_id related to the devices waiting for a place in the sensors network. As soon as a place will be available, the first pending invitation_id of the queue will be used to invite the related device waiting to be part of the net.
        :param devices_with_ready_audio_to_send_list: list - it's a list of the devices that already notified they have an audio ready to send.
               Whenever a device send a notification to the RaspberryPy, it's device_id will be pushed into the list. Vice versa, whenever an audio has been successfully received, the related device_id will be popped out by the list.
        :param start_index_ready_devices_list: int - it's the start_index for devices_with_ready_audio_to_send_list that identifies a group pf device_ids to request for an audio.
                E.g., if we have: devices_with_ready_audio_to_send_list = [ID1, ID2, ID3, ID4, ID5, ID6, ID7, ID8, ID9, ID10, ID11, ID12, ID13, ID14], then start_index will have value:
                - 0 --> so that we can request an audio in parallel (4 parallel requests since the RaspberryPy is quad-core) to the devices with ids: ID1, ID2, ID3, ID4, since this group starts from index 0;
                - 4 --> so that we can request an audio in parallel (4 parallel requests since the RaspberryPy is quad-core) to the devices with ids: ID5, ID6, ID7, ID8, since this group starts from index 4;
                - 8 --> so that we can request an audio in parallel (4 parallel requests since the RaspberryPy is quad-core) to the devices with ids: ID9, ID10, ID11, ID12, since this group starts from index 8;
                - 12 --> so that we can request an audio in parallel (4 parallel requests since the RaspberryPy is quad-core) to the devices with ids: ID13, ID14, since this group starts from index 12;
                So, after every parallel audio requests (made of max requests) and the consecutive audio saving, start_index_ready_devices_list will be increased for the next parallel audio request, and so on (after all the IDs have been interrogated, the variable will be set to 0)
        :param threads_communication_id_dict: dict - dictionary containing the communication_ids created in the multithreaded audio request. Since these must be used also in the audio/ack topic, it's better to save them for that occurrence as a class field
        :param thread_audio_data_topic_key_dict: dict - dictionary containing the audio_data_topic_keys created in the multithreaded audio request. Since these must be used also in the audio/payload topic, it's better to save them for that occurrence as a class field
        :param go_on_with_audio_requests_dict: dict - dict of key:values pairs like threadx:True/False, where if True it means that it is possible to make the next request for that thread (the previous request has been successfully handled), otherwise not if False
                When it's time to make the next multithreaded requests, every value of the dict must be True, so a check on these is necessary.
        :param max_resending_request: int - the maximum number of audio resend request allowed (if an audio is not well received for the max time allowed, then the problem must be ignored sending an ack:"ok" to the device)
        :param counters_resend_request_done_per_thread_dict: dict - a dictionary containing, for every of the four threads, the counters of the resending requests, in order to take trace of it if a counter becomes >= than the max_resending_request number
        :param current_device_id_handled_by_thread_dict: dict - a dictionary containing as keys the four threads (e.g. thread1) and for each of them the value is the current device_id handled by that thread at the moment (assigned during the audio request phase)
        """

        self.device_model = device_model
        self.device_id = device_id
        self.num_current_devices = num_current_devices
        self.num_max_allowed_devices = num_max_allowed_devices
        self.num_max_pending_subscription_requests = num_max_pending_subscription_requests
        self.status = status
        self.ip_address = ip_address
        self.start_time_sending_audio_request = start_time_sending_audio_request
        self.max_waiting_time_for_receiving_audio = max_waiting_time_for_receiving_audio
        self.audio_data_topic_key = audio_data_topic_key
        self.communication_id = communication_id
        self.recording_frequency = recording_frequency
        self.devices_infos_dict = devices_infos_dict
        self.invitation_id_list = invitation_id_list # e.g. [invitation_id1, invitation_id2, invitation_id3, invitation_id4, ...]
        self.devices_with_ready_audio_to_send_list = devices_with_ready_audio_to_send_list # e.g. [device_id13, device_id4, device_id5, device_id11, ...]
        self.start_index_ready_devices_list = start_index_ready_devices_list #e.g. can be one of the following: 0/4/8/12/16/20....see the example above in the comment section of this class
        self.threads_communication_id_dict = threads_communication_id_dict # e.g. {"thread1": communication_id1, "thread2": communication_id2, "thread3": communication_id3, "thread4": communication_id4}
        self.thread_audio_data_topic_key_dict = thread_audio_data_topic_key_dict # e.g. {"thread1": topic_key1, "thread2": topic_key2, "thread3": topic_key3, "thread4": topic_key4}
        self.go_on_with_audio_requests_dict = go_on_with_audio_requests_dict # e.g. {"thread1": True, "thread2": False, "thread3": True, "thread4": False}
        self.max_resending_request = max_resending_request # e.g. 3 (3 attempts to resend an audio)
        self.counters_resend_request_done_per_thread_dict = counters_resend_request_done_per_thread_dict # e.g. {"thread1": 0, "thread2": 2, "thread3": 0, "thread4": 1}
        self.current_device_id_handled_by_thread_dict = current_device_id_handled_by_thread_dict

    def update_devices_infos_dict(self, json_received_infos):
        device_id = json_received_infos["device_id"]
        device_infos = {
        "device_model" : json_received_infos["device_model"],
        "location" : json_received_infos["location"],
        "battery_level" : json_received_infos["battery_level"],
        "status" : json_received_infos["status"],
        "ip_address" : json_received_infos["ip_address"],
        "enable_audio_record" : json_received_infos["enable_audio_record"],
        "last_record_date" : json_received_infos["last_record_date"],
        "recording_frequency" : json_received_infos["recording_frequency"]
        }
        self.devices_infos_dict.update({device_id: device_infos})




        self.devices_infos_dict.update({})
    def update_current_device_id_handled_by_thread_dict(self, current_device_id_handled_by_thread_dict):
        self.current_device_id_handled_by_thread_dict = current_device_id_handled_by_thread_dict

    def update_threads_communication_id_dict(self, key, value):
        self.threads_communication_id_dict.update({key: value})

    def update_thread_audio_data_topic_key_dict(self, key, value):
        self.thread_audio_data_topic_key_dict.update({key: value})

    def get_audio_data_topic_key(self):
        return self.audio_data_topic_key

    def get_recording_frequency(self):
        return self.recording_frequency

    def set_audio_data_topic_key(self, audio_data_topic_key):
        self.audio_data_topic_key = audio_data_topic_key

    def get_start_time_sending_audio_request(self):
        return self.start_time_sending_audio_request

    def set_start_time_sending_audio_request(self):
        return self.start_time_sending_audio_request

    def get_max_waiting_time_for_receiving_audio(self):
        return self.max_waiting_time_for_receiving_audio

    def set_max_waiting_time_for_receiving_audio(self, max_waiting_time_for_receiving_audio):
        self.max_waiting_time_for_receiving_audio = max_waiting_time_for_receiving_audio

    def set_go_on_with_audio_requests(self, go_on_with_audio_requests):
        self.go_on_with_audio_requests = go_on_with_audio_requests

    def get_go_on_with_audio_requests(self):
        return self.go_on_with_audio_requests

    def return_go_on_with_audio_request_flag(self):
        counter_trues = 0
        for key in self.go_on_with_audio_requests_dict:
            if self.go_on_with_audio_requests_dict[key] is True:
                counter_trues += 1
        if counter_trues == len(self.go_on_with_audio_requests_dict):
            go_on_with_the_request = True
        else:
            go_on_with_the_request = False
        return go_on_with_the_request


    device_id = infos_obtained_from_esp32["device_id"]
    del infos_obtained_from_esp32["device_id"] #deleting the key:value with key "device_id", so that it remains the all dict without it, and we can use it as a 'value' in the update
    RaspberryPy.devices_infos_dict.update({device_id:infos_obtained_from_esp32})
    '''
    UPDATING THE DEVICE INFOS :
    {device_id: 
        {
            device_model,
            location,
            battery_level,
            status,
            ip_address,
            enable_audio_record,
            last_record_date,
            recording_frequency}
        }
    '''

def remove_pending_device_from_invitation_id_list(infos_obtained_from_esp32):
    raspberrypy3B.invitation_id_list.remove(infos_obtained_from_esp32["invitation_id"])  # removing the new added ESP32 from the invitation_id_list


def generate_random_code(length=20):
    # Create a pool of uppercase, lowercase letters and digits
    characters = string.ascii_letters + string.digits
    # Generate a random code by selecting random characters from the pool
    random_code = ''.join(random.choice(characters) for _ in range(length))
    return random_code

def is_wav_file(data):
    # Check if the first four bytes are the b'RIFF' and the next four are b'WAVE'
    return len(data) > 8 and data[0:4] == b'RIFF' and data[8:12] == b'WAVE'

def publish_positive_response_to_subscription_request(topic, invitation_id, device_id, recording_frequency):
    positive_response = {"device_id":device_id, "invitation_id": invitation_id, "Connection_permission":"allowed", "recording_frequency":recording_frequency}
    positive_response = json.dumps(positive_response)
    print(f"Sending: {positive_response}")
    client.publish(topic, positive_response)

def publish_negative_response_to_subscription_request(topic, invitation_id):
    negative_response = {"invitation_id": invitation_id, "Connection_permission":"denied"}
    negative_response = json.dumps(negative_response)
    print(f"Sending: {negative_response}")
    client.publish(topic, negative_response)

def publish_audio_request(topic, device_id, communication_id, audio_data_topic_key):

    request = {"device_id": device_id, "audio_data_topic_key": audio_data_topic_key, "communication_id": communication_id}
    request = json.dumps(request)
    print(f"Sending: {request}")
    client.publish(topic, request)

    '''
    topic: audio/request
    ESP32                                                                                                RASPBERRYPY
    _____                                                                                                ______
    # 1      {device_id:<ID>, audio_data_topic_key:<topic_key>, communication_id:<ID>}<---------------------

    Dynamic topic: audio/payload<key>
    ESP32                                                                                                RASPBERRYPY
    _____                                                                                                ______
    # 1      -------------------------------------------------------------------------------------------->audio_content
    '''

def request_audio_thread1(client, RaspberryPy, device_id):
    audio_data_topic_key = generate_random_code(20)
    client.subscribe("audio/payload"+audio_data_topic_key)
    RaspberryPy.thread_audio_data_topic_key_dict.update({"thread1", audio_data_topic_key})
    communication_id = generate_random_code(20)
    RaspberryPy.update_threads_communication_id_dict("thread1", communication_id)
    publish_audio_request("audio/request", device_id, communication_id, audio_data_topic_key)

def request_audio_thread2(client, RaspberryPy, device_id):
    audio_data_topic_key = generate_random_code(20)
    client.subscribe("audio/payload" + audio_data_topic_key)
    RaspberryPy.thread_audio_data_topic_key_dict.update({"thread2", audio_data_topic_key})
    communication_id = generate_random_code(20)
    RaspberryPy.update_threads_communication_id_dict("thread2", communication_id)
    publish_audio_request("audio/request", device_id, communication_id, audio_data_topic_key)

def request_audio_thread3(client, RaspberryPy, device_id):
    audio_data_topic_key = generate_random_code(20)
    client.subscribe("audio/payload" + audio_data_topic_key)
    RaspberryPy.thread_audio_data_topic_key_dict.update({"thread3", audio_data_topic_key})
    communication_id = generate_random_code(20)
    RaspberryPy.update_threads_communication_id_dict("thread3", communication_id)
    publish_audio_request("audio/request", device_id, communication_id, audio_data_topic_key)

def request_audio_thread4(client, RaspberryPy, device_id):
    audio_data_topic_key = generate_random_code(20)
    client.subscribe("audio/payload" + audio_data_topic_key)
    RaspberryPy.thread_audio_data_topic_key_dict.update({"thread4", audio_data_topic_key})
    communication_id = generate_random_code(20)
    RaspberryPy.update_threads_communication_id_dict("thread4", communication_id)
    publish_audio_request("audio/request", device_id, communication_id, audio_data_topic_key)

def ack_audio_received_thread(client, RaspberryPy, thread_number, ack):
    # audio/ack:
    # 3      {communication_id:<ID>, audio_data_topic_key:<topic_key>, ack:"resend"}<-----------------------------------
    # 3      {communication_id:<ID>, ack:"ok"}<-----------------------------------
    if ack == "resend":
        # 3      {communication_id:<ID>, audio_data_topic_key:<topic_key>, ack:"resend"}<-----------------------------------
        audio_data_topic_key = generate_random_code(20) # must be different than the previous one
        topic_where_send_the_audio = "audio/payload" + audio_data_topic_key
        client.subscribe(topic_where_send_the_audio)
        key_audio_data_topic_key = "audio_data_topic_key_" + str(thread_number) # where thread_number can be one between 1/2/3/4 for the related threads
        key_communication_id = "communication_id_" + str(thread_number) # where thread_number can be one between 1/2/3/4 for the related threads
        RaspberryPy.thread_audio_data_topic_key_dict.update({key_audio_data_topic_key, audio_data_topic_key})
        communication_id = RaspberryPy.threads_communication_id_dict[key_communication_id]
        message = {"communication_id": communication_id, "audio_data_topic_key": audio_data_topic_key, "ack": ack}
        message = json.dumps(message)
        print(f"Sending: {message}")
        client.publish("audio/ack", message)
    elif ack == "ok":
        # 3      {communication_id:<ID>, ack:"ok"}<-----------------------------------
        key_communication_id = "communication_id_" + str(thread_number) # where thread_number can be one between 1/2/3/4 for the related threads
        communication_id = RaspberryPy.threads_communication_id_dict[key_communication_id]
        message = {"communication_id": communication_id, "ack": ack}
        message = json.dumps(message)
        print(f"Sending: {message}")
        client.publish("audio/ack", message)

def request_audio_multithreading(RaspberryPy):

    devices_with_ready_audio_to_send_list = RaspberryPy.devices_with_ready_audio_to_send_list
    mod = len(devices_with_ready_audio_to_send_list) % 4
    if mod == 0:
        last_start_index = len(devices_with_ready_audio_to_send_list) - 4   # e.g., for a number of devices that is exactly multiple of 4 --> 12 devices --> last_start_index = 12 - 4 = 8
    else:
        last_start_index = round(len(devices_with_ready_audio_to_send_list)/4)*4    # e.g., for less than 4 devices --> 3 devices --> last_start_index = round(3/4)*4 = 0*4 = 0
                                                                                    # e.g., for more than 4 devices but less than a multiple of 4 --> 14 devices --> last_start_index = round(14/4)*4 = 3*4 = 12
    return_true_at_last_audio_requests_group = False
    last_group_dim = len(devices_with_ready_audio_to_send_list) - last_start_index
    # for i in devices_with_ready_audio_to_send_list:
    if (RaspberryPy.start_index_ready_devices_list != last_start_index) or (RaspberryPy.start_index_ready_devices_list == last_start_index and last_group_dim == 4):
        start_index = RaspberryPy.start_index_ready_devices_list
        end_index = start_index + 3 # where 3 = current_group_dim-1 = 4-1, where 4 is for sure the dimension of the current group since we are not at the last_start_index now
        temp_ready_dev_list = devices_with_ready_audio_to_send_list[start_index : end_index]
        request_audio_thread1(client, RaspberryPy,temp_ready_dev_list[0])
        request_audio_thread2(client, RaspberryPy,temp_ready_dev_list[1])
        request_audio_thread3(client, RaspberryPy,temp_ready_dev_list[2])
        request_audio_thread4(client, RaspberryPy,temp_ready_dev_list[3])
        temp_thread_device_is_dict = {"thread1": temp_ready_dev_list[0], "thread2": temp_ready_dev_list[1], "thread3": temp_ready_dev_list[2], "thread4": temp_ready_dev_list[3]}
        RaspberryPy.update_current_device_id_handled_by_thread_dict(temp_thread_device_is_dict)
        RaspberryPy.start_index_ready_devices_list+= 4

    elif RaspberryPy.start_index_ready_devices_list >= last_start_index: # last audio requests to be done
        start_index = RaspberryPy.start_index_ready_devices_list
        end_index = start_index + last_group_dim-1
        temp_ready_dev_list = devices_with_ready_audio_to_send_list[start_index : end_index]
        if last_group_dim == 1: # if there is only one device_id in the last_group
            request_audio_thread1(client, RaspberryPy,temp_ready_dev_list[0])
            temp_thread_device_is_dict = {"thread1": temp_ready_dev_list[0]}
            RaspberryPy.update_current_device_id_handled_by_thread_dict(temp_thread_device_is_dict)
            RaspberryPy.start_index_ready_devices_list = 0 # since we are done doing audio requests, the start_index can be moved to the first element for the next complete series of audio requests (at the next frequency period)
            # RaspberryPy.start_index_ready_devices_list+= 1
        if last_group_dim == 2: # if there are only two device_ids in the last_group
            request_audio_thread1(client, RaspberryPy,temp_ready_dev_list[0])
            request_audio_thread2(client, RaspberryPy,temp_ready_dev_list[1])
            temp_thread_device_is_dict = {"thread1": temp_ready_dev_list[0], "thread2": temp_ready_dev_list[1]}
            RaspberryPy.update_current_device_id_handled_by_thread_dict(temp_thread_device_is_dict)
            RaspberryPy.start_index_ready_devices_list = 0 # since we are done doing audio requests, the start_index can be moved to the first element for the next complete series of audio requests (at the next frequency period)
            # RaspberryPy.start_index_ready_devices_list+= 2
        if last_group_dim == 3: # if there are only three device_ids in the last_group
            request_audio_thread1(client, RaspberryPy,temp_ready_dev_list[0])
            request_audio_thread2(client, RaspberryPy,temp_ready_dev_list[1])
            request_audio_thread3(client, RaspberryPy,temp_ready_dev_list[2])
            temp_thread_device_is_dict = {"thread1": temp_ready_dev_list[0], "thread2": temp_ready_dev_list[1], "thread3": temp_ready_dev_list[2]}
            RaspberryPy.update_current_device_id_handled_by_thread_dict(temp_thread_device_is_dict)
            RaspberryPy.start_index_ready_devices_list = 0 # since we are done doing audio requests, the start_index can be moved to the first element for the next complete series of audio requests (at the next frequency period)
            # RaspberryPy.start_index_ready_devices_list+= 3
        return_true_at_last_audio_requests_group = True

    return return_true_at_last_audio_requests_group

def check_time_elapsed_for_receiving_audio(RaspberryPy):
    current_time = time.time()
    if current_time - RaspberryPy.start_time_sending_audio_request > RaspberryPy.max_waiting_time_for_receiving_audio:

# Callback function when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print(f"Connected to broker at {broker} with result code {rc}")
    for topic, qos in topics:
        client.subscribe(topic)
        print(f"Subscribed to topic: {topic}")


# Callback function when a message is received.
def on_message(client, userdata, msg, raspberrypy3B):
    print(f"Message received on topic {msg.topic}: {msg.payload.decode('utf-8')}")

    if msg.topic == "connect_to_sensors_net/request":
        # 1 -------------------------------------------------------------------------------------------->{invitation_id: <ID>, infos: {infos}}}  (infos without 'device_id')
        # 2 {invitation_id:<ID>, device_id:<ID>, Connection_permission:<allowed/denied>, recording_frequency:<freq>}<----------------------------------------------
        # 3 -------------------------------------------------------------------------------------------->{invitation_id: <ID>, infos: {infos}}}  (infos with 'device_id')
        json_received = msg.payload.decode('utf-8')
        json_received = json.loads(json_received)  # Convert the payload to a Python dictionary
        try:
            if raspberrypy3B.num_current_devices < raspberrypy3B.num_max_allowed_devices: # the device's request to be part of the sensors network will be accepted
                if json_received["infos"]["device_id"] is None:
                    # 1 ------------------------------->{invitation_id: <ID>, infos: {infos}}}  (infos without 'device_id')
                    assigned_device_id = generate_random_code(20)
                    assigned_recording_frequency = raspberrypy3B.get_recording_frequency() # e.g. 5 min between two consecutive audio recordings (300 sec)
                    publish_positive_response_to_subscription_request("connect_to_sensors_net/request", json_received["invitation_id"], assigned_device_id, assigned_recording_frequency)
                else:
                    # 3 ------------------------------->{invitation_id: < ID >, infos: {infos}}}  (infos with 'device_id')
                    raspberrypy3B.update_devices_infos_dict(json_received["infos"]) # adding the infos of the new ESP32 added
            elif raspberrypy3B.num_current_devices >= raspberrypy3B.num_max_allowed_devices: # the device's request to be part of the sensors network will not be accepted since there is no space
                if len(raspberrypy3B.invitation_id_list) < raspberrypy3B.num_max_pending_subscription_requests: #the ESP32 can be enlisted in the pending queue to be part of the sensors network
                    publish_negative_response_to_subscription_request("connect_to_sensors_net/request", json_received["invitation_id"])
                else: #the ESP32 cannot even be enlisted in the pending queue to be part of the sensors network, since the queue is full
                    pass # no infos about the sensors network should be returned to the ESP32

        except json.JSONDecodeError:
            print("Received payload is not valid JSON.")
            print("ESP32: problem receiving the package for the topic 'connect_to_sensors_net/request'")

    if msg.topic == "connect_to_sensors_net/a_place_is_available":
        # waiting for an invitation by the RaspberryPy itself, at the topic: "connect_to_sensors_net/a_place_is_available", using the same Device.communication_id made up for the previous topic
        # 1      {invitation_id:<ID>, device_id:<ID>, recording_frequency<freq>}<----------------------------------------------
        # 2      -------------------------------------------------------------------------------------------->{invitation_id:<ID>,infos:{infos}}}
        json_received = msg.payload.decode('utf-8')
        json_received = json.loads(json_received)  # Convert the payload to a Python dictionary
        try:
            if json_received["invitation_id"] in raspberrypy3B.invitation_id_list: # we now can add the ESP32 that was waiting for a place in the sensors network
                # 2      -------------------------------------------------------------------------------------------->{invitation_id:<ID>,infos:{infos}}}
                raspberrypy3B.update_devices_infos_dict(json_received["infos"]) # adding the infos of the new ESP32 added
                remove_pending_device_from_invitation_id_list(json_received["invitation_id"]) # removing the new added ESP32 from the invitation_id_list
        except json.JSONDecodeError:
            print("Received payload is not valid JSON.")
            print("ESP32: problem receiving the package for the topic 'connect_to_sensors_net/a_place_is_available'")

    # elif msg.topic == "audio/request": this must be handled by the ESP32
    # 1      {device_id:<ID>, audio_data_topic_key:<topic_key>, communication_id:<ID>}<--------------------------------------------------

    # elif msg.topic == "audio/ack":
    # 1      {communication_id:<ID>, audio_data_topic_key:<topic_key>, ack:ok/resend}<-----------------------------------


    if msg.topic == "audio/payload" + raspberrypy3B.thread_audio_data_topic_key_dict["thread1"]:
        # audio/request:
        #1      {device_id:<ID>, audio_data_topic_key:<topic_key>, communication_id:<ID>}<----------------------------------
        # audio/payload<key>:
        #2      (if ack == resend it will be used the new dynamic_topic_key for resending the audio)----->{audio_content}

        if is_wav_file(msg.payload):
            os.makedirs("received_files/thread1", exist_ok=True)
            device_id = raspberrypy3B.current_device_id_handled_by_thread_dict["thread1"]
            device_model = raspberrypy3B.devices_infos_dict[device_id]["device_model"]
            location = raspberrypy3B.devices_infos_dict[device_id]["location"]
            last_record_date = raspberrypy3B.devices_infos_dict[device_id]["last_record_date"]
            file_name = device_model + "_" + location + "_" + last_record_date
            with open("received_audio_files/thread1/" + file_name + ".wav", "wb") as audio_file:
                audio_file.write(msg.payload)
            print("WAV audio file saved as ", "received_audio_files/thread1/" + file_name + ".wav")
            go_on_with_audio_request_thread1 = True
        else:
            print("Received file is not a valid .wav file.")
            go_on_with_audio_request_thread1 = False

        # audio/ack:
        #3      {communication_id:<ID>, audio_data_topic_key:<topic_key>, ack:"resend"}<-----------------------------------
        #3      {communication_id:<ID>, ack:"ok"}<-----------------------------------
        if go_on_with_audio_request_thread1 is True:
            ack_audio_received_thread(client, raspberrypy3B, 1, "ok")
        else:  # ask to resend the audio
            if raspberrypy3B.counters_resend_request_done_per_thread_dict["thread1"] < raspberrypy3B.max_resending_request:
                ack_audio_received_thread(client, raspberrypy3B, 1, "resend")
            else:  # the max attempts to resend the audio were made, thus for not doing infinite attempts, just ignore the fact that the audio is not ok --> signaling that everything is ok (since we don't care anymore)
                ack_audio_received_thread(client, raspberrypy3B, 1, "ok")
                go_on_with_audio_request_thread1 = True

        with lock:
            raspberrypy3B.go_on_with_audio_requests_dict.update({"thread1": go_on_with_audio_request_thread1})
        '''
        Aspetto di ricevere l'audio
            -se passa troppo tempo senza riceverlo --> ack:resend, new topic_key
            -se lo ricevo controllo che sia valido:
                -se è valido --> ack:ok
                -se non è valido --> ack:resend, new topic_key # questa operazione va fatta solo un tot di volte per non rischiare di ripeterla all'infinito
        '''

    if msg.topic == "audio/payload" + raspberrypy3B.thread_audio_data_topic_key_dict["thread2"]:
        # audio/request:
        # 1      {device_id:<ID>, audio_data_topic_key:<topic_key>, communication_id:<ID>}<----------------------------------
        # audio/payload<key>:
        # 2      (if ack == resend it will be used the new dynamic_topic_key for resending the audio)----->{audio_content}

        if is_wav_file(msg.payload):
            os.makedirs("received_files/thread2", exist_ok=True)
            device_id = raspberrypy3B.current_device_id_handled_by_thread_dict["thread2"]
            device_model = raspberrypy3B.devices_infos_dict[device_id]["device_model"]
            location = raspberrypy3B.devices_infos_dict[device_id]["location"]
            last_record_date = raspberrypy3B.devices_infos_dict[device_id]["last_record_date"]
            file_name = device_model + "_" + location + "_" + last_record_date
            with open("received_audio_files/thread2/" + file_name + ".wav", "wb") as audio_file:
                audio_file.write(msg.payload)
            print("WAV audio file saved as ", "received_audio_files/thread2/" + file_name + ".wav")
            go_on_with_audio_request_thread2 = True
        else:
            print("Received file is not a valid .wav file.")
            go_on_with_audio_request_thread2 = False

        # audio/ack:
        # 3      {communication_id:<ID>, audio_data_topic_key:<topic_key>, ack:"resend"}<-----------------------------------
        # 3      {communication_id:<ID>, ack:"ok"}<-----------------------------------
        if go_on_with_audio_request_thread2 is True:
            ack_audio_received_thread(client, raspberrypy3B, 2, "ok")
        else:  # ask to resend the audio
            if raspberrypy3B.counters_resend_request_done_per_thread_dict["thread2"] < raspberrypy3B.max_resending_request:
                ack_audio_received_thread(client, raspberrypy3B, 2, "resend")
            else:  # the max attempts to resend the audio were made, thus for not doing infinite attempts, just ignore the fact that the audio is not ok --> signaling that everything is ok (since we don't care anymore)
                ack_audio_received_thread(client, raspberrypy3B, 2, "ok")
                go_on_with_audio_request_thread2 = True

        with lock:
            raspberrypy3B.go_on_with_audio_requests_dict.update({"thread2": go_on_with_audio_request_thread2})
        '''
        Aspetto di ricevere l'audio
            -se passa troppo tempo senza riceverlo --> ack:resend, new topic_key
            -se lo ricevo controllo che sia valido:
                -se è valido --> ack:ok
                -se non è valido --> ack:resend, new topic_key # questa operazione va fatta solo un tot di volte per non rischiare di ripeterla all'infinito
        '''

    if msg.topic == "audio/payload" + raspberrypy3B.thread_audio_data_topic_key_dict["thread3"]:
        # audio/request:
        # 1      {device_id:<ID>, audio_data_topic_key:<topic_key>, communication_id:<ID>}<----------------------------------
        # audio/payload<key>:
        # 2      (if ack == resend it will be used the new dynamic_topic_key for resending the audio)----->{audio_content}

        if is_wav_file(msg.payload):
            os.makedirs("received_files/thread3", exist_ok=True)
            device_id = raspberrypy3B.current_device_id_handled_by_thread_dict["thread3"]
            device_model = raspberrypy3B.devices_infos_dict[device_id]["device_model"]
            location = raspberrypy3B.devices_infos_dict[device_id]["location"]
            last_record_date = raspberrypy3B.devices_infos_dict[device_id]["last_record_date"]
            file_name = device_model + "_" + location + "_" + last_record_date
            with open("received_audio_files/thread3/" + file_name + ".wav", "wb") as audio_file:
                audio_file.write(msg.payload)
            print("WAV audio file saved as ", "received_audio_files/thread3/" + file_name + ".wav")
            go_on_with_audio_request_thread3 = True
        else:
            print("Received file is not a valid .wav file.")
            go_on_with_audio_request_thread3 = False

        # audio/ack:
        # 3      {communication_id:<ID>, audio_data_topic_key:<topic_key>, ack:"resend"}<-----------------------------------
        # 3      {communication_id:<ID>, ack:"ok"}<-----------------------------------
        if go_on_with_audio_request_thread3 is True:
            ack_audio_received_thread(client, raspberrypy3B, 3, "ok")
        else:  # ask to resend the audio
            if raspberrypy3B.counters_resend_request_done_per_thread_dict["thread3"] < raspberrypy3B.max_resending_request:
                ack_audio_received_thread(client, raspberrypy3B, 3, "resend")
            else:  # the max attempts to resend the audio were made, thus for not doing infinite attempts, just ignore the fact that the audio is not ok --> signaling that everything is ok (since we don't care anymore)
                ack_audio_received_thread(client, raspberrypy3B, 3, "ok")
                go_on_with_audio_request_thread3 = True

        with lock:
            raspberrypy3B.go_on_with_audio_requests_dict.update({"thread3": go_on_with_audio_request_thread3})
        '''
        Aspetto di ricevere l'audio
            -se passa troppo tempo senza riceverlo --> ack:resend, new topic_key
            -se lo ricevo controllo che sia valido:
                -se è valido --> ack:ok
                -se non è valido --> ack:resend, new topic_key # questa operazione va fatta solo un tot di volte per non rischiare di ripeterla all'infinito
        '''


    if msg.topic == "audio/payload" + raspberrypy3B.thread_audio_data_topic_key_dict["thread4"]:
        # audio/request:
        # 1      {device_id:<ID>, audio_data_topic_key:<topic_key>, communication_id:<ID>}<----------------------------------
        # audio/payload<key>:
        # 2      (if ack == resend it will be used the new dynamic_topic_key for resending the audio)----->{audio_content}

        if is_wav_file(msg.payload):
            os.makedirs("received_files/thread4", exist_ok=True)
            device_id = raspberrypy3B.current_device_id_handled_by_thread_dict["thread4"]
            device_model = raspberrypy3B.devices_infos_dict[device_id]["device_model"]
            location = raspberrypy3B.devices_infos_dict[device_id]["location"]
            last_record_date = raspberrypy3B.devices_infos_dict[device_id]["last_record_date"]
            file_name = device_model + "_" + location + "_" + last_record_date
            with open("received_audio_files/thread4/" + file_name + ".wav", "wb") as audio_file:
                audio_file.write(msg.payload)
            print("WAV audio file saved as ", "received_audio_files/thread4/" + file_name + ".wav")
            go_on_with_audio_request_thread4 = True
        else:
            print("Received file is not a valid .wav file.")
            go_on_with_audio_request_thread4 = False

        # audio/ack:
        # 3      {communication_id:<ID>, audio_data_topic_key:<topic_key>, ack:"resend"}<-----------------------------------
        # 3      {communication_id:<ID>, ack:"ok"}<-----------------------------------
        if go_on_with_audio_request_thread4 is True:
            ack_audio_received_thread(client, raspberrypy3B, 4, "ok")
        else:  # ask to resend the audio
            if raspberrypy3B.counters_resend_request_done_per_thread_dict["thread4"] < raspberrypy3B.max_resending_request:
                ack_audio_received_thread(client, raspberrypy3B, 4, "resend")
            else: # the max attempts to resend the audio were made, thus for not doing infinite attempts, just ignore the fact that the audio is not ok --> signaling that everything is ok (since we don't care anymore)
                ack_audio_received_thread(client, raspberrypy3B, 4, "ok")
                go_on_with_audio_request_thread4 = True

        with lock:
            raspberrypy3B.go_on_with_audio_requests_dict.update({"thread4": go_on_with_audio_request_thread4})
        '''
        Aspetto di ricevere l'audio
            -se passa troppo tempo senza riceverlo --> ack:resend, new topic_key
            -se lo ricevo controllo che sia valido:
                -se è valido --> ack:ok
                -se non è valido --> ack:resend, new topic_key # questa operazione va fatta solo un tot di volte per non rischiare di ripeterla all'infinito
        '''

    if msg.topic == "get_infos":
        # 1      {device_id:<ID>, communication_id:<ID>}<--------------------------------------------------
        json_received = msg.payload.decode('utf-8')
        json_received = json.loads(json_received)  # Convert the payload to a Python dictionary
        try:
            if json_received["device_id"] == Device.device_id:
                Device.update_communication_id(json_received["communication_id"])
                publish_infos_with_communication_id("get_infos", Device, json_received["communication_id"])
        except json.JSONDecodeError:
            print("Received payload is not valid JSON.")
            print("ESP32: problem receiving the package for the topic 'get_infos'")

    if msg.topic == "sleep_mode":
        # 1      {device_id:<ID>, communication_id:<ID>}<--------------------------------------------------
        json_received = msg.payload.decode('utf-8')
        json_received = json.loads(json_received)  # Convert the payload to a Python dictionary
        try:
            if json_received["device_id"] == Device.device_id:
                Device.update_communication_id(json_received["communication_id"])
                publish_infos_with_communication_id("get_infos", Device, json_received["communication_id"])
                Device.update_enable_audio_record("no")
                time.sleep(2)
                topics_to_unsuscribe = [
                    ("connect_to_sensors_net/request", 0),  # -->   <--
                    ("audio/request", 0),  # <--
                    ("audio/ack", 0),  # <--
                    ("sleep_mode", 0),  # <--   -->publish_status
                ]
                for topic, qos in topics_to_unsuscribe:
                    client.unsubscribe(topic)
                    print(f"Unsubscribed from topic: {topic}")
        except json.JSONDecodeError:
            print("Received payload is not valid JSON.")
            print("ESP32: problem receiving the package for the topic 'sleep_mode'")

    if msg.topic == "awake_device":
        # 1      {device_id:<ID>, communication_id:<ID>}<--------------------------------------------------
        json_received = msg.payload.decode('utf-8')
        json_received = json.loads(json_received)  # Convert the payload to a Python dictionary
        try:
            if json_received["device_id"] == Device.device_id:
                Device.update_communication_id(json_received["communication_id"])
                publish_infos_with_communication_id("get_infos", Device, json_received["communication_id"])
                Device.update_enable_audio_record("yes")
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
        except json.JSONDecodeError:
            print("Received payload is not valid JSON.")
            print("ESP32: problem receiving the package for the topic 'awake_device'")

    if msg.topic == "disconnection_from_broker":
        # 1      {device_id:<ID>, communication_id:<ID>}<--------------------------------------------------
        json_received = msg.payload.decode('utf-8')
        json_received = json.loads(json_received)  # Convert the payload to a Python dictionary
        try:
            if json_received["device_id"] == Device.device_id:
                Device.update_communication_id(json_received["communication_id"])
                publish_infos_with_communication_id("get_infos", Device, json_received["communication_id"])
                client.disconnect()  # Disconnect from the mqtt broker
                Device.update_enable_audio_record("no")
        except json.JSONDecodeError:
            print("Received payload is not valid JSON.")
            print("ESP32: problem receiving the package for the topic 'disconnection_from_broker'")

    if msg.topic == "reset":
        # 1      {device_id:<ID>, communication_id:<ID>}<--------------------------------------------------
        json_received = msg.payload.decode('utf-8')
        json_received = json.loads(json_received)  # Convert the payload to a Python dictionary
        try:
            if json_received["device_id"] == Device.device_id:
                Device.update_communication_id(json_received["communication_id"])
                publish_infos_with_communication_id("get_infos", Device, json_received["communication_id"])
                reset_device()  # Resetting the device, so everything will be restarted from the beginning
        except json.JSONDecodeError:
            print("Received payload is not valid JSON.")
            print("ESP32: problem receiving the package for the topic 'reset'")

    if msg.topic == "set_recording_frequency":
        # 1      {device_id:<ID>, communication_id:<ID>}<--------------------------------------------------
        json_received = msg.payload.decode('utf-8')
        json_received = json.loads(json_received)  # Convert the payload to a Python dictionary
        try:
            if json_received["device_id"] == Device.device_id:
                Device.update_recording_frequency(json_received["recording_frequency"])
                Device.update_communication_id(json_received["communication_id"])
                publish_infos_with_communication_id("get_infos", Device, json_received["communication_id"])
        except json.JSONDecodeError:
            print("Received payload is not valid JSON.")
            print("ESP32: problem receiving the package for the topic 'awake_device'")


'''
# Callback function when a message is received.
def on_message(client, userdata, msg):
    print(f"Message received on topic {msg.topic}, checking file type...")

    if msg.topic == "audio/wav":
        if is_wav_file(msg.payload):
            os.makedirs("received_files", exist_ok=True)
            with open("received_files/received_audio.wav", "wb") as audio_file:
                audio_file.write(msg.payload)
            print("WAV audio file saved as 'received_files/received_audio.wav'!")
        else:
            print("Received file is not a valid .wav file.")

    elif msg.topic == "audio/mp3":
        # Handle MP3 messages if necessary
        os.makedirs("received_files", exist_ok=True)
        with open("received_files/received_audio.mp3", "wb") as audio_file:
            audio_file.write(msg.payload)
        print("MP3 audio file saved as 'received_files/received_audio.mp3'!")

    elif msg.topic == "test/topic":
        # Handle messages from other/topic if necessary
        print(f"Message from other/topic: {msg.payload.decode('utf-8')}")
'''


def run_cpp_executable(message):
    # Here, 'your_cpp_program' is the name/path of your C++ executable
    try:
        # You can pass the message as an argument to the C++ program
        result = subprocess.run(['./your_cpp_program', message], capture_output=True, text=True)
        print("C++ Program Output:", result.stdout)
        if result.returncode != 0:
            print("C++ Program Error:", result.stderr)
    except FileNotFoundError:
        print("C++ program not found. Make sure the executable path is correct.")
    except Exception as e:
        print("An error occurred while running the C++ program:", str(e))

##############################################################

SR = 8000
N_FFT = 256
HOP_LEN = int(N_FFT / 2)
input_shape = (129, 1251, 1)
##############################################################

class CustomRandom:
    def __init__(self, seed):
        # Ensure m is set properly
        self.m = 2 ** 32  # 4294967296
        self.a = 1664525
        self.c = 1013904223
        self.seed = seed

    def custom_random(self):
        self.seed = (self.a * self.seed + self.c) % self.m
        return self.seed

    def custom_randint(self, low, high):
        if low >= high:
            raise ValueError("low must be less than high")
        return low + (self.custom_random() % (high - low))


def parse_config_file(file_path):
    audio_folder = None
    save_folder = None
    tflite_model_path = None

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if line.startswith('#audio_folder'):
                audio_folder = next(file).strip()  # Read next line for the path
            elif line.startswith('#save_folder'):
                save_folder = next(file).strip()  # Read next line for the path
            elif line.startswith('#tflite_model_path'):
                tflite_model_path = next(file).strip()  # Read next line for the path

    return audio_folder, save_folder, tflite_model_path


# Function to preprocess audio files
def preprocess_audio(audio_path, sr=8000, n_fft=256, hop_length=128):
    #    print("__ enter def preprocess_audio(audio_path, sr=8000, n_fft=256, hop_length=128):")

    # Load audio file
    data, _ = librosa.load(audio_path, sr=sr)
    #    print("\n\n=============================DATA LOADED=================================")
    #    print("data.dtype", data.dtype)
    #    print("data.shape", data.shape)
    # Adjust NumPy print options to show more elements
    # np.set_printoptions(threshold=np.inf)  # Show all elements
    np.set_printoptions(threshold=100)  # Show all elements
    #print(data)
    #    print("First 100 values:", data[:100])
    #    print("Last 100 values:", data[-100:])
    # Reset the print options if necessary
    np.set_printoptions(threshold=1000)  # Reset to the default value
    #    print("==================================================================\n\n")

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # Random roll as in training
    #    print("-----------------------------------------ROLL------------------------------------------")
    seed = 2018
    custom_rng = CustomRandom(seed)
    #    print("Python Seed:", seed)

    # Generate random roll amount in a separate instance
    roll_amount = custom_rng.custom_randint(0, len(data))
    #    print("Roll amount (Python):", roll_amount)

    data = np.roll(data, -roll_amount)

    np.set_printoptions(threshold=100)  # Show all elements
    #    print("rolled_vector: ")
    # print(data)
    #    print("First 100 values:", data[:100])
    #    print("Last 100 values:", data[-100:])
    # Reset the print options if necessary
    np.set_printoptions(threshold=1000)  # Reset to the default value

    #    print("data.dtype:", data.dtype)
    #    print("-----------------------------------------------------------------------------------")
    # data = np.roll(data, random.randint(0, len(data))) #sostituito dal codice subito sopra
    #data = roll(data, seed)

    # Replicate if len(data) < 20 seconds
    if len(data) < 20 * sr:
        data = np.repeat(data, int(20 * sr / len(data) + 1))
    # Compute the STFT
    stft_result = librosa.stft(data, n_fft=n_fft, hop_length=hop_length)

    #    print("\n\n=============================DATA AFTER STFT=================================")
    # np.set_printoptions(threshold=np.inf)  # Show all elements
    #    print("stft_result.dtype", stft_result.dtype)
    #    print("stft_result.size", stft_result.size)
    #    print("stft_result.shape", stft_result.shape)

    # for saving values on file
    #outputFile = open('stft_output.txt', 'w')
    #print(stft_result, file=outputFile)
    #outputFile.close()

    #print(stft_result)
    #    print("First 100 values:", stft_result[:100])
    #    print("Last 100 values:", stft_result[-100:])
    # np.set_printoptions(threshold=1000)  # Reset to the default value
    #    print("==================================================================\n\n")


    # Convert amplitude to decibels
    data = librosa.amplitude_to_db(abs(stft_result))
    #    print("\n\n=============================DATA AFTER AMPLITUDE TO dB=================================")
    #    print("data.dtype", data.dtype)
    #    print("data.size", data.size)
    #    print("data.shape", data.shape)


    # print(data)
    #    print("First 100 values:", data[:100])
    #    print("Last 100 values:", data[-100:])
    #    print("==================================================================\n\n")
    # Process the data

    data = data[:, :1251]  # Take first 1251 columns
    data = np.flipud(data)  # Flip data

    #    print("\n\n=============================DATA AFTER VERTICAL FLIP=================================")
    #    print("data.dtype", data.dtype)
    # print(data)
    #    print("First 100 values:", data[:100])
    #    print("Last 100 values:", data[-100:])
    #    print("==================================================================\n\n")

    data = np.expand_dims(data, axis=-1) # viene aggiunta una dimensione, che consiste nel riporre ogni numero di ogni riga di array, in un array a se (cioè ogni riga avrà 1251 array, ognuno contenente un numero)
    data = np.expand_dims(data, axis=0)  # This creates shape (1, 129, 1251, 1)
    #    print('data.shape: ', data.shape)
    #    print("data.dtype", data.dtype)
    #    print("\n\n=============================DATA AFTER DIMENSIONS EXPANSION (FINAL DATA TO BE LOADED INTO THE MODEL)=================================")
    # print(data)
    #    print("First 100 values:", data[:100])
    #    print("Last 100 values:", data[-100:])
    #    print("==================================================================\n\n")
    return data

# Function to perform inference on audio files
def print_spectrogram(audio_path,output_file_name):
    #    print("__ enter def print_spectrogram(audio_path,output_file_name):")

    # Preprocess the audio
    #    print("//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////")
    processed_audio = preprocess_audio(audio_path)
    #    print("processed_audio.dtype", processed_audio.dtype)

    # Save the spectrogram image; ensure it's 2D
    # Save the spectrogram image; ensure it's 4D
    #plt.imsave(output_file_name, processed_audio[0, :, :, 0], cmap='viridis')  # Save only 2D for visualization, but keep for inference
    #    print('processed_audio.shape: ', processed_audio.shape)  # Should print (1, 129, 1251, 1)
    #    print('processed_audio: ', processed_audio)

    # Assume processed_audio is your Numpy array with shape (1, 129, 1251, 1)

    # Remove the last dimension if needed for saving as an array
    processed_audio_to_save = processed_audio[0]  # shape (129, 1251, 1)
    # Save the array as .npy file
    np.save(output_file_name, processed_audio_to_save)

    # Save the array as a binary file
    extension = ".npy"
    output_binary_file, extension = os.path.splitext(output_file_name)
    output_binary_file = output_binary_file + '.bin'
    #    print('output_binary_file: ', output_binary_file)
    with open(output_binary_file, 'wb') as f:
        processed_audio_to_save.tofile(f)  # Write the raw byte data to the binary file

    # Load the binary file back into a NumPy array
    # Specify the dtype (data type) and count (shape of the array)
    data_reloaded = np.fromfile(output_binary_file, dtype=np.float32)  # Adjust dtype if necessary
    # Optionally, if you know the shape, reshape it
    #data_reloaded = data_reloaded.reshape((5,))  # Adjust shape according to the original data
    #    print(data_reloaded)  # This should print: [1 2 3 4 5]
    #    print("data_reloaded.shape: ", data_reloaded.shape)
    #    print("___________________________________________________________________________________________________________________________________")



def preprocess_and_obtain_npy_spectrograms_from_audio_folder_thread1(audio_folder):
    # Classify each audio file in the folder
    from pathlib import Path
    for root, _, files in os.walk(audio_folder):
        for file in files:
            if file.endswith(".wav"):
                # output_file_name = str(Path(file).stem) + '.png'
                output_file_name = str(Path(file).stem) + '.npy'
                print('output_file_name: ', output_file_name)
                save_folder = audio_folder
                # print_spectrogram(os.path.join(root, file), output_file_name)
                print_spectrogram(os.path.join(root, file), save_folder + output_file_name)

def preprocess_and_obtain_npy_spectrograms_from_audio_folder_thread2(audio_folder):
    # Classify each audio file in the folder
    from pathlib import Path
    for root, _, files in os.walk(audio_folder):
        for file in files:
            if file.endswith(".wav"):
                # output_file_name = str(Path(file).stem) + '.png'
                output_file_name = str(Path(file).stem) + '.npy'
                print('output_file_name: ', output_file_name)
                save_folder = audio_folder
                # print_spectrogram(os.path.join(root, file), output_file_name)
                print_spectrogram(os.path.join(root, file), save_folder + output_file_name)

def preprocess_and_obtain_npy_spectrograms_from_audio_folder_thread3(audio_folder):
    # Classify each audio file in the folder
    from pathlib import Path
    for root, _, files in os.walk(audio_folder):
        for file in files:
            if file.endswith(".wav"):
                # output_file_name = str(Path(file).stem) + '.png'
                output_file_name = str(Path(file).stem) + '.npy'
                print('output_file_name: ', output_file_name)
                save_folder = audio_folder
                # print_spectrogram(os.path.join(root, file), output_file_name)
                print_spectrogram(os.path.join(root, file), save_folder + output_file_name)

def preprocess_and_obtain_npy_spectrograms_from_audio_folder_thread4(audio_folder):
    # Classify each audio file in the folder
    from pathlib import Path
    for root, _, files in os.walk(audio_folder):
        for file in files:
            if file.endswith(".wav"):
                # output_file_name = str(Path(file).stem) + '.png'
                output_file_name = str(Path(file).stem) + '.npy'
                print('output_file_name: ', output_file_name)
                save_folder = audio_folder
                # print_spectrogram(os.path.join(root, file), output_file_name)
                print_spectrogram(os.path.join(root, file), save_folder + output_file_name)




# audio_file_path = 'treevibes/field/field/train/clean/folder_25/F_20200524144643_1.wav'  # Replace with your audio file path
# Define folder paths
# clean_folder = 'treevibes/field/field/train/clean/folder_25'
# infested_folder = 'treevibes/field/field/train/infested/folder_19'


# Get the current working directory
current_directory = os.getcwd()

#    print("Current Working Directory:", current_directory)

# Use the function to get the paths
file_path = current_directory+'/setting_file.txt'  # Change this to the actual file path
audio_folder, save_folder, tflite_model_path = parse_config_file(file_path)

# Output the results
#    print("Audio Folder:", audio_folder)
#    print("Save Folder:", save_folder)
#    print("TFLite Model Path:", tflite_model_path)




audio_files = []
target_names = ['clean', 'infested']  # Define your target class names



######################################################################################

# Create an MQTT client instance
client = mqtt.Client()

# Attach the callback functions
client.on_connect = on_connect
client.on_message = on_message

# Connect to the broker
client.connect(broker, port, 60)

# Start the loop in a separate thread to process network traffic and dispatch callbacks
client.loop_start()
time.sleep(10)
# Publish a message to the topic
client.publish(topic, "Hello, MQTT from Raspberry!")
print(f"Message published to '{topic}'")

# Keep the script running to listen for incoming messages
last_audio_request_publish_time = time.time()
publish_interval = 60  # seconds

# Create a threading lock
lock = threading.Lock()

try:
    while True:
        current_time = time.time()

        # Check if it's time to request the audios
        if current_time - last_audio_request_publish_time >= publish_interval and len(raspberrypy3B.devices_with_ready_audio_to_send_list) != 0:

            return_true_at_last_audio_requests_group = False
            return_true_at_last_audio_requests_group = request_audio_multithreading(raspberrypy3B)

            while return_true_at_last_audio_requests_group is False:
                # check if the previous requests have been completely handled, thus we can go on doing the next parallel requests
                with lock:
                    go_on_with_the_request = return_go_on_with_audio_request_flag(raspberrypy3B)
                if go_on_with_the_request is True:
                    return_true_at_last_audio_requests_group = request_audio_multithreading(raspberrypy3B)

            # Update the last published time
            last_audio_request_publish_time = current_time

        preprocess_and_obtain_npy_spectrograms_from_audio_folder_thread1("received_audio_files/thread1/")
        preprocess_and_obtain_npy_spectrograms_from_audio_folder_thread2("received_audio_files/thread2/")
        preprocess_and_obtain_npy_spectrograms_from_audio_folder_thread3("received_audio_files/thread3/")
        preprocess_and_obtain_npy_spectrograms_from_audio_folder_thread4("received_audio_files/thread4/")



        # Process MQTT events (including handling callbacks)
        client.loop()  # This will process network events, including any incoming messages and callbacks

        # Optional: Sleep for a short time to reduce CPU usage
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Exiting...")
finally:
    client.loop_stop()  # Stop the loop gracefully
    client.disconnect()  # Disconnect from the broker

########################################################################################


# Classify each audio file in the folder
from pathlib import Path
for root, _, files in os.walk(audio_folder):
    for file in files:
        if file.endswith(".wav"):
            #output_file_name = str(Path(file).stem) + '.png'
            output_file_name = str(Path(file).stem) + '.npy'
            print('output_file_name: ', output_file_name)

            #print_spectrogram(os.path.join(root, file), output_file_name)
            print_spectrogram(os.path.join(root, file), save_folder+output_file_name)

end_time = time.time()    # Record the end time

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")






if __name__ == "__main__":
    raspberrypy3B = RaspberryPy(
        device_model="RaspberryPy 3B",
        device_id="Rpi3B",
        num_current_devices=0,
        num_max_allowed_devices=6,
        devices_infos_dict={},
        status="active",
        ip_address="192.168.1.200",
        start_time_sending_audio_request=None,
        max_waiting_time_for_receiving_audio=5,
        audio_data_topic_key=None,
        communication_id=None,
        invitation_id_list=[]
    )


    print(raspberrypy3B)







