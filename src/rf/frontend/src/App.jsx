import { useEffect, useRef } from 'react';
import GainControl from './components/GainControl';
import CenterFreqControl from './components/CenterFreqControl';
import { startFFTPlot } from './fftPlot';

function App() {
  const chartRef = useRef();
  const chartRef2 = useRef();

  useEffect(() => {
    if (chartRef.current && chartRef2.current) {
      startFFTPlot(chartRef.current, chartRef2.current, "ws://localhost:8767");
    }
    // if (chartRef2.current) {
    //   startFFTPlot(chartRef2.current, "ws://localhost:8766");
    // }
  }, []);

  return (
    <div style={{ margin: 0, padding: 0, display: 'flex', flexDirection: 'column', height: '90vh', width: '90vw' }}>
      <div style={{ margin: 0, background: '#111', color: '#ccc', height: '70%', width: '100vw' }}>
        <div ref={chartRef} style={{ height: '100%', width: '100%' }} />
      </div>
      <div style={{ margin: 0, background: '#111', color: '#ccc', height: '20%', width: '100vw' }}>
        <div ref={chartRef2} style={{ height: '100%', width: '100%' }} />
      </div>
      <div style={{ margin: 0, background: '#111', color: '#ccc', height: '10%', width: '100vw' }}>
        <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center', height: '100%' }}>
          <GainControl />
          <CenterFreqControl />
        </div>
      </div>
    </div>
  );
}

export default App;
