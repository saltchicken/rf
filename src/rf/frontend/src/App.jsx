import { useEffect, useRef } from 'react';
import GainControl from './components/GainControl';
import CenterFreqControl from './components/CenterFreqControl';
import FFTPlot from './components/FFTPlot';

function App() {
  return (
    <div style={{ margin: 0, padding: 0, display: 'flex', flexDirection: 'column', height: '90vh', width: '90vw' }}>
      <div style={{ margin: 0, background: '#111', color: '#ccc', height: '70%', width: '100vw' }}>
        <FFTPlot websocketUrl="ws://localhost:8767" />
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
