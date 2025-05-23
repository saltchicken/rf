import zmq
import struct

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5000")  # Replace with actual host and port
socket.setsockopt_string(zmq.SUBSCRIBE, "samples")  # Subscribe to 'samples' topic

while True:
    topic, msg = socket.recv_multipart()
    length = struct.unpack('!I', msg[:4])[0]
    data = msg[4:]
    print(f"Received {len(data)} bytes on topic '{topic.decode()}'")

