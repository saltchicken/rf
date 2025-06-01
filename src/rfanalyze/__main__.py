import argparse
import configparser
import asyncio

from .client import (
    Reader,
    ReaderListener,
    ReaderRecorder,
    ReaderFFT,
    ReaderConstellation,
)

from pathlib import Path

config_dir = f"{Path(__file__).parent}/config"


def open_config_file():
    import os, sys

    try:
        if sys.platform == "darwin":
            os.system(f"open {config_dir}/config.ini")
        elif sys.platform == "nt":
            os.startfile(f"{config_dir}/config.ini")
        else:
            os.system(f"xdg-open {config_dir}/config.ini")
    except Exception as e:
        print(f"Failed to open config file: {e}")


def get_args(command=None):
    config = configparser.ConfigParser()
    config.read(f"{config_dir}/config.ini")

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--host",
        type=str,
        default=config["Network"]["HOST"],
        help="Host to connect to.",
    )
    parent_parser.add_argument(
        "--port",
        type=int,
        default=config["Network"]["PORT"],
        help="Port number to listen on.",
    )
    parent_parser.add_argument(
        "--sample_rate",
        type=float,
        default=config["Processing"]["SAMPLE_RATE"],
        help="Sample rate.",
    )
    parent_parser.add_argument(
        "--freq_offset",
        type=float,
        default=config["Demodulation"]["FREQ_OFFSET"],
        help="Frequency offset for signal shifting (in Hz).",
    )
    parent_parser.add_argument(
        "--chunk_size",
        type=int,
        default=config["Processing"]["CHUNK_SIZE"],
        help="Chunk size for processing samples.",
    )

    # Main parser
    parser = argparse.ArgumentParser(description="FM receiver and demodulator.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser: listen
    listen_parser = subparsers.add_parser(
        "listen", parents=[parent_parser], help="Listen to FM broadcast"
    )

    # Subparser: record
    record_parser = subparsers.add_parser(
        "record", parents=[parent_parser], help="Record FM to file"
    )
    record_parser.add_argument(
        "--duration",
        type=int,
        default=config["Processing"]["DURATION"],
        help="Recording duration in seconds",
    )
    record_parser.add_argument("--output_filename", type=str, help="Output file name")

    fft_parser = subparsers.add_parser(
        "fft", parents=[parent_parser], help="Real-time FFT visualization"
    )
    fft_parser.add_argument(
        "--chunks_per_frame",
        type=int,
        default=config["FFT"]["CHUNKS_PER_FRAME"],
        help="Chunks to accumalate per FFT calculation",
    )
    fft_parser.add_argument(
        "--decimation_factor",
        type=int,
        default=config["FFT"]["DECIMATION_FACTOR"],
        help="Decimation factor for FFT calculation",
    )

    constellation_parser = subparsers.add_parser(
        "constellation", parents=[parent_parser], help="Constellation visualization"
    )

    edit_parser = subparsers.add_parser("edit", help="Edit config file")
    # edit_parser.add_argument('file', type=str, help='Config file to edit')

    command_parser = subparsers.add_parser(
        "command", parents=[parent_parser], help="Send command to FM receiver"
    )
    command_parser.add_argument("setting", type=str, help="Setting to modify")
    command_parser.add_argument("value", type=str, help="Value to set")

    if command:
        return parser.parse_args([command])
    else:
        return parser.parse_args()


async def run():
    args = get_args()

    if args.command == "record":
        reader_recorder = ReaderRecorder(args)
        # await reader_recorder.set_setting('gain', 30)
        # settings = await reader_recorder.get_current_settings()
        # print(settings)
        await reader_recorder.run()
    elif args.command == "listen":
        reader_listener = ReaderListener(args)
        receive_task = asyncio.create_task(reader_listener.receive_samples())
        listen_task = asyncio.create_task(reader_listener.listen_sample())
        await asyncio.gather(listen_task, receive_task)
    elif args.command == "fft":
        reader_fft = ReaderFFT(args)
        await reader_fft.run()
    elif args.command == "edit":
        open_config_file()
    elif args.command == "constellation":
        reader_constellation = ReaderConstellation(args)
        await reader_constellation.run()
    elif args.command == "command":
        reader = Reader(args)
        response = await reader.set_setting(args.setting, args.value)
        print(response)
    else:
        raise ValueError(f"Unknown command: {args.command}")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
