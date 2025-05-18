
from classes import SDRRecorder

def main():
    # Initialize the SDR
    recorder = SDRRecorder(sample_rate=1e6)
    
    try:
        # Record from first FM station (92.1 MHz)
        samples = recorder.record(center_freq=92.0e6, duration_seconds=4, freq_offset=3e5)
        recorder.fm_demodulate_and_save(samples, "station1.wav")
        
        # Record from second FM station (95.5 MHz)
        samples = recorder.record(center_freq=95.5e6, duration_seconds=4, freq_offset=3e5)
        recorder.fm_demodulate_and_save(samples, "station2.wav")
        
    finally:
        # Always close the SDR properly
        recorder.close()

if __name__ == "__main__":
    main()
