import asyncio
from rfanalyze import ReaderRecorder, get_args

async def run_recorder_instance(index, freq_offset):
    args = get_args('record')
    args.index = index
    args.freq_offset = freq_offset
    args.output_filename = f'output_{index}.wav'

    reader_recorder = ReaderRecorder(args)
    results = await reader_recorder.run()


async def main():
    await asyncio.gather(
        run_recorder_instance(0, 0),
        run_recorder_instance(1, 1e5),
        run_recorder_instance(2, 2e5),
        run_recorder_instance(3, 3e5),
    )


if __name__ == '__main__':
    asyncio.run(main())
