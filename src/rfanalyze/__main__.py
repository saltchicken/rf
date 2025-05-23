import argparse
import configparser
import asyncio

from .client import ReaderListener, ReaderRecorder
from .fft import RealTimeFFTVisualizer
from .server import Receiver

from pathlib import Path
config_dir = f'{Path(__file__).parent}/config'

def open_config_file():
    import os, sys
    try:
        if sys.platform == 'darwin':
            os.system(f'open {config_dir}/config.ini')
        elif sys.platform == 'nt':
            os.startfile(f'{config_dir}/config.ini')
        else:
            os.system(f'xdg-open {config_dir}/config.ini')
    except Exception as e:
        print(f'Failed to open config file: {e}')

def get_args(virtual=False):
    config = configparser.ConfigParser()
    config.read(f'{config_dir}/config.ini')

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--host', type=str, default=config['Network']['HOST'], help='Host to connect to.')
    parent_parser.add_argument('--port', type=int, default=config['Network']['PORT'], help='Port number to listen on.')
    parent_parser.add_argument('--sample_rate', type=float, default=config['Processing']['SAMPLE_RATE'], help='Sample rate.')
    parent_parser.add_argument('--freq_offset', type=float, default=config['Demodulation']['FREQ_OFFSET'], help='Frequency offset for signal shifting (in Hz).')
    parent_parser.add_argument('--chunk_size', type=int, default=config['Processing']['CHUNK_SIZE'], help='Chunk size for processing samples.')

    # Main parser
    parser = argparse.ArgumentParser(description='FM receiver and demodulator.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Subparser: listen
    listen_parser = subparsers.add_parser('listen', parents=[parent_parser], help='Listen to FM broadcast')

    # Subparser: record
    record_parser = subparsers.add_parser('record', parents=[parent_parser], help='Record FM to file')
    record_parser.add_argument('--duration', type=int, default=config['Processing']['DURATION'], help='Recording duration in seconds')

    fft_parser = subparsers.add_parser('fft', parents=[parent_parser], help='Real-time FFT visualization')
    fft_parser.add_argument('--chunks_per_frame', type=int, default=config['FFT']['CHUNKS_PER_FRAME'], help='Chunks to accumalate per FFT calculation')
    fft_parser.add_argument('--decimation_factor', type=int, default=config['FFT']['DECIMATION_FACTOR'], help='Decimation factor for FFT calculation')

    receiver_parser = subparsers.add_parser('server', parents=[parent_parser], help='Run FM receiver as a server')
    receiver_parser.add_argument('--center_freq', type=float, default=config['Server']['CENTER_FREQ'], help='Center frequency.')
    receiver_parser.add_argument('--buffer_size', type=int, default=config['Server']['BUFFER_SIZE'], help='Buffer size.')
    receiver_parser.add_argument('--gain', type=float, default=config['Server']['GAIN'], help='Gain.')

    edit_parser = subparsers.add_parser('edit', help='Edit config file')
    # edit_parser.add_argument('file', type=str, help='Config file to edit')

    if virtual:
        return parser.parse_args(['test'])
    else:
        return parser.parse_args()

async def run():
    args = get_args()

    if args.command == 'record':
        reader_recorder = ReaderRecorder(args)
        receive_task = asyncio.create_task(reader_recorder.receive_samples())
        record_task = asyncio.create_task(reader_recorder.record_sample(duration_seconds = args.duration))
        await asyncio.gather(record_task, receive_task)
    elif args.command == 'listen':
        reader_listener = ReaderListener(args)
        receive_task = asyncio.create_task(reader_listener.receive_samples())
        listen_task = asyncio.create_task(reader_listener.listen_sample())
        await asyncio.gather(listen_task, receive_task)
    elif args.command == 'fft':
        fft_visualizer = RealTimeFFTVisualizer(args)
        fft_visualizer.run()
    elif args.command == 'server':
        receiver = Receiver(args)
        await receiver.stream_samples()
    elif args.command == 'edit':
        open_config_file()
    else:
        raise ValueError(f"Unknown command: {args.command}")

def main():
    asyncio.run(run())

if __name__ == '__main__':
    main()
