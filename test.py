from multiprocessing import Pool
import asyncio
from rfanalyze import ReaderRecorder, get_args


def run_recorder_instance(index_freq):
    index, freq_offset = index_freq

    async def runner():
        args = get_args('record')
        args.index = index
        args.freq_offset = freq_offset
        args.output_filename = f'output_{index}.wav'

        reader_recorder = ReaderRecorder(args)
        reader_recorder.save = False
        return await reader_recorder.run()

    return asyncio.run(runner())


def main():
    freq_offsets = [0, 1e5, 2e5, 3e5, 4e5, 5e5]
    args_list = [(i, f) for i, f in enumerate(freq_offsets)]

    with Pool(processes=len(args_list)) as pool:
        results = pool.map(run_recorder_instance, args_list)

    for index, result in enumerate(results):
        print(f"Process {index} result: {result}")


if __name__ == '__main__':
    main()
