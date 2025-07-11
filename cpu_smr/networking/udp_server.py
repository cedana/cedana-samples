#!/usr/bin/env python3

import signal
import socket
import sys
import threading


def handle_exit(signum, frame):
    sys.exit(1)


signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

# Define server address and port
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 12345

# Create a UDP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind the socket to the address and port
server_socket.bind((SERVER_HOST, SERVER_PORT))

print(f'UDP server listening on {SERVER_HOST}:{SERVER_PORT}')

# Function to handle incoming messages from clients
def handle_client(message, client_address):
    print(f'Received message from {client_address}: {message.decode()}')

    # Respond to the client
    response = f'Message received: {message.decode()}'
    server_socket.sendto(response.encode(), client_address)


# Listen for incoming messages and spawn threads for each
while True:
    message, client_address = server_socket.recvfrom(
        1024
    )  # Buffer size 1024 bytes
    # Create a new thread for each client
    threading.Thread(
        target=handle_client, args=(message, client_address)
    ).start()
