import GainControl from './components/GainControl';
import CenterFreqControl from './components/CenterFreqControl';
import FFTPlot from './components/FFTPlot';

function App() {
  return (
    <div style={{ margin: 0, padding: 0, display: 'flex', flexDirection: 'column', height: '100vh', width: '100vw' }}>
      <div style={{ margin: 0, background: '#111', color: '#ccc', height: '100%', width: '100%' }}>
        <FFTPlot websocketUrl="ws://localhost:8767" />
        <div style={{
          position: 'absolute',
          top: '10px',
          right: '10px',
          width: '30%',
          height: '30%',
          zIndex: 10,
          border: '1px solid #333',
          borderRadius: '4px',
          overflow: 'hidden'
        }}>
          <FFTPlot websocketUrl="ws://localhost:8768" />
        </div>
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
