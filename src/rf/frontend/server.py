import asyncio
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from rf import ReaderFFT, ReaderListener
from pydantic import BaseModel
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.readerFFT = await ReaderFFT.create("10.0.0.5", 5000)
    app.state.readerListener = await ReaderListener.create("10.0.0.5", 5000)
    print(app.state.readerFFT.center_freq)
    app.state.reader_task = asyncio.create_task(app.state.readerFFT.run())
    app.state.reader_task2 = asyncio.create_task(app.state.readerListener.run())
    print("ReaderFFT tasks started.")

    yield

    # Shutdown
    async def cancel_task(task, name):
        if task:
            print(f"Cancelling {name}...")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                print(f"{name} cancelled cleanly.")

    await cancel_task(app.state.reader_task, "ReaderFFT")
    await cancel_task(app.state.reader_task2, "ReaderListener")


app = FastAPI(lifespan=lifespan)

# CORS and static file setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

assets_path = Path(__file__).parent / "dist/assets"
dist_path = Path(__file__).parent / "dist"
app.mount("/assets", StaticFiles(directory=assets_path), name="assets")


@app.get("/")
def serve_index():
    return FileResponse(dist_path / "index.html")


class SettingPayload(BaseModel):
    setting: str
    value: float


@app.post("/api/set-setting")
async def button_click(setting: SettingPayload):
    response = await app.state.readerFFT.set_setting(setting.setting, setting.value)
    if setting.setting == "center_freq":
        await app.state.readerFFT.publisher.message_queue.put(setting.value)
    return {"message": f"{response}"}


class XValue(BaseModel):
    x: float


@app.post("/api/selected_x")
async def receive_x(value: XValue):
    print(f"Received x value: {value.x}")
    # offset = round(value.x - app.state.readerFFT.center_freq)
    # app.state.readerFFT.freq_offset = offset
    # app.state.readerListener.freq_offset = offset
    app.state.readerFFT.freq_offset = value.x
    app.state.readerListener.freq_offset = value.x
    return {"status": "ok", "x_received": value.x}
