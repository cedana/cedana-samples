import socket


def handle_exit(signum, frame):
    sys.exit(1)


signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

# Define server address and port
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 12345

# Create a UDP socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Message to send
message = 'Hello, server!'

# Send the message to the server
client_socket.sendto(message.encode(), (SERVER_HOST, SERVER_PORT))

# Receive the response from the server
response, _ = client_socket.recvfrom(1024)
print(f'Server response: {response.decode()}')

# Close the socket
client_socket.close()
