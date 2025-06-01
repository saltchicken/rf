import Plotly from 'https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.26.0/+esm';

export function startWaterfallPlot(container, websocketUrl) {
  const N_FREQ_BINS = 1024;     // Number of frequency bins (FFT bins)
  const MAX_TIME_SLICES = 40; // Number of time slices visible at once

  let fftData = [];
  for (let i = 0; i < MAX_TIME_SLICES; i++) {
    var blank = Array(N_FREQ_BINS).fill(0);
    fftData.push(blank);
  }

  // Initialize x-axis (time slices)
  let timeLabels = [];
  for (let i = 0; i < MAX_TIME_SLICES; i++) {
    timeLabels.push(i+1);
  }

  // Initialize y-axis (frequency bins)
  let freqLabels = [];
  for (let i = 0; i < N_FREQ_BINS; i++) {
    freqLabels.push(i+1);  // you can map to actual frequencies if you want
  }

  let layout = {
    // title: 'Real-time FFT Heatmap',
    plot_bgcolor: '#1e1e1e',
    paper_bgcolor: '#1e1e1e',
    xaxis: { title: 'Freq bins' },
    yaxis: { title: 'Time (slices)' },
  };

  let data = [{
    z: fftData,
    y: timeLabels,
    x: freqLabels,
    type: 'heatmap',
    colorscale: 'Viridis',
    zmin: 0,
    zmax: 30,
    showscale: false,
  }];

  // Plotly.newPlot('heatmap', data, layout, {responsive: true});
  Plotly.newPlot(container, data, layout);

  const ws = new WebSocket(websocketUrl);
  ws.binaryType = 'arraybuffer';

  ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      const json = JSON.parse(event.data);
      if (json.type === "init") {
        console.log("init received")
        freqLabels = Array.from(new Float32Array(json.data));
      }
    } else if (event.data instanceof ArrayBuffer) {
      let mags = Array.from(new Float32Array(event.data));
      // let mags = Array(N_FREQ_BINS).fill(0);
      // console.log(fftData)
      fftData.push(mags);
      if (fftData.length > MAX_TIME_SLICES) {
        fftData.shift();
      }
      Plotly.update(container, { x: [freqLabels], z: [fftData] } );
    }
  };

  ws.onopen = () => {
    ws.send(JSON.stringify({ type: "init", data: "waterfall" })); };


}
