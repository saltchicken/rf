import asyncio
import json
import websockets
import zmq
import zmq.asyncio


class Publisher:
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.queue = asyncio.Queue(maxsize=1)
        self.message_queue = asyncio.Queue(maxsize=1)
        self.server_task = asyncio.create_task(self.start_server())

    async def handler(self, ws):
        async def receive_and_forward_message():
            while True:
                msg = await self.message_queue.get()
                await ws.send(json.dumps({"type": "update", "data": msg}))

        async def receive_and_forward():
            while True:
                msg = await self.queue.get()
                await ws.send(msg)

        async def message_from_client():
            async for message in ws:
                message = json.loads(message)
                if message.get("type") == "init":
                    data = message.get("data")
                    print(f"Received message: {data}")
                    await ws.send(json.dumps({"type": "init", "data": data}))
                    # subscriber.setsockopt_string(zmq.SUBSCRIBE, message)

        try:
            await asyncio.gather(
                receive_and_forward(),
                receive_and_forward_message(),
                message_from_client(),
            )
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            print("Handle any cleanup that is necessary")

    async def start_server(self):
        async with websockets.serve(self.handler, self.host, self.port):
            print(f"WebSocket server running on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(start_server())
