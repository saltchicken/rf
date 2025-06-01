import asyncio
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from rf import ReaderFFT, ReaderListener
from pydantic import BaseModel

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static assets (JS, CSS, etc.)
app.mount("/assets", StaticFiles(directory="dist/assets"), name="assets")


# Serve index.html at root
@app.get("/")
def serve_index():
    return FileResponse("dist/index.html")


@app.on_event("startup")
async def start_readerfft():
    app.state.readerFFT = await ReaderFFT.create("10.0.0.5", 5000)
    app.state.readerListener = await ReaderListener.create("10.0.0.5", 5000)
    app.state.reader_task = asyncio.create_task(app.state.readerFFT.run())
    app.state.reader_task2 = asyncio.create_task(app.state.readerListener.run())
    # settings = await readerFFT.get_current_settings()
    await app.state.readerFFT.update_settings()
    await app.state.readerListener.update_settings()
    # print(settings)
    print("ReaderFFT tasks started.")


async def cancel_task(task, name):
    if task:
        print(f"Cancelling {name}...")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            print(f"{name} cancelled cleanly.")


@app.on_event("shutdown")
async def shutdown_readerfft():
    await cancel_task(app.state.reader_task, "ReaderFFT")
    await cancel_task(app.state.reader_task2, "ReaderListener")


class SettingPayload(BaseModel):
    setting: str
    value: float


@app.post("/api/set-setting")
async def button_click(setting: SettingPayload):
    response = await app.state.readerFFT.set_setting(setting.setting, setting.value)
    if setting.setting == "center_freq":
        # print("Setting Center Freq")
        # readerFFT.center_freq = setting.value
        # readerListener.center_freq = setting.value
        await app.state.readerFFT.publisher.message_queue.put(setting.value)
    return {"message": f"{response}"}


class XValue(BaseModel):
    x: float


@app.post("/api/selected_x")
async def receive_x(value: XValue):
    print(f"Received x value: {value.x}")
    offset = round(value.x - app.state.readerFFT.center_freq)
    app.state.readerFFT.freq_offset = offset
    app.state.readerListener.freq_offset = offset
    # Do something with the x value (e.g. store, trigger something)
    return {"status": "ok", "x_received": value.x}
