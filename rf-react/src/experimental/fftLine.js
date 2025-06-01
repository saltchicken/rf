import Plotly from 'https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.26.0/+esm';

export function startFFTPlot(container, websocketUrl = "ws://localhost:8765") {
  let freqs = [];
  let mags = [];

  const layout = {
    title: { text: 'Real-Time FFT', font: { color: '#ccc' } },
    plot_bgcolor: '#1e1e1e',
    paper_bgcolor: '#1e1e1e',
    xaxis: { title: 'Frequency (Hz)', color: '#ccc', gridcolor: '#222' },
    yaxis: { title: 'Magnitude', range: [0, 100], autorange: false, color: '#ccc', gridcolor: '#222' }
  };

  const data = [{
    x: [],
    y: [],
    mode: 'lines',
    line: { color: 'cyan' },
    name: 'Magnitude'
  }];

  Plotly.newPlot(container, data, layout, { responsive: true });

  const ws = new WebSocket(websocketUrl);
  ws.binaryType = "arraybuffer";

  ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      const json = JSON.parse(event.data);
      if (json.type === "init") {
        freqs = Array.from(new Float32Array(json.data));
      }
    } else if (event.data instanceof ArrayBuffer) {
      mags = Array.from(new Float32Array(event.data));
      Plotly.update(container, { x: [freqs], y: [mags] });
    }
  };

  ws.onopen = () => {
    ws.send(JSON.stringify({ type: "init", data: "plotly" }));
  };
}

