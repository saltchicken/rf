import multiprocessing
import asyncio
from rfanalyze import ReaderRecorder, get_args


def run_recorder_instance(index, freq_offset):
    async def runner():
        args = get_args('record')
        args.index = index
        args.freq_offset = freq_offset
        args.output_filename = f'output_{index}.wav'

        reader_recorder = ReaderRecorder(args)
        await reader_recorder.run()

    asyncio.run(runner())


def main():
    processes = []
    freq_offsets = [0, 1e5, 2e5, 3e5, 4e5, 5e5]

    for index, freq in enumerate(freq_offsets):
        p = multiprocessing.Process(target=run_recorder_instance, args=(index, freq))
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')  # Ensure compatibility on macOS/Windows
    main()
