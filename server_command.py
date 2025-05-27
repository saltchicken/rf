import zmq

ctx = zmq.Context()
socket = ctx.socket(zmq.REQ)
# TODO: Change this to the host in config
socket.connect("tcp://10.0.0.5:5001")  # REP port = PUB port + 1

socket.send_json({"gain": 45.0, "center_freq": 98.1e6})
response = socket.recv_json()
print(response)

