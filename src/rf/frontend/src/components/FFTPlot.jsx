import { useEffect, useRef } from 'react';
import Plotly from 'https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.26.0/+esm';

function FFTPlot({ websocketUrl = "ws://localhost:8765" }) {
  const fftContainerRef = useRef(null);
  const waterfallContainerRef = useRef(null);

  useEffect(() => {
    if (!fftContainerRef.current || !waterfallContainerRef.current) return;

    const containerFFT = fftContainerRef.current;
    const containerWaterfall = waterfallContainerRef.current;

    // Constants
    const N_FREQ_BINS = 1024;
    const MAX_TIME_SLICES = 40;
    const sample_rate = 2000000;
    const center_freq = sample_rate / 2;
    const bin_width = sample_rate / N_FREQ_BINS;

    // Initialize data
    let freqs = Array.from({ length: N_FREQ_BINS }, (_, i) =>
      (center_freq - sample_rate / 2 + i * bin_width) - sample_rate / 2
    );

    let mags = [];
    let fftData = Array(MAX_TIME_SLICES).fill().map(() => Array(N_FREQ_BINS).fill(0));
    let timeLabels = Array.from({ length: MAX_TIME_SLICES }, (_, i) => i + 1);
    let isDragging = false;

    // Plot layouts and data
    const FFTLayout = {
      margin: { l: 50, r: 50, t: 50, b: 0 },
      title: { text: 'Real-Time FFT', font: { color: '#ccc' } },
      plot_bgcolor: '#1e1e1e',
      paper_bgcolor: '#1e1e1e',
      xaxis: { zeroline: false, showticklabels: false, nticks: 10, color: '#ccc', gridcolor: '#222' },
      yaxis: { zeroline: false, title: 'Magnitude', range: [-50, 50], autorange: false, color: '#ccc', gridcolor: '#222' }
    };

    // Initialize plots
    Plotly.newPlot(containerFFT, [{
      x: [],
      y: [],
      mode: 'lines',
      line: { color: 'cyan' },
      name: 'Magnitude'
    }], FFTLayout, { responsive: true, staticPlot: true });

    Plotly.newPlot(containerWaterfall, [{
      z: fftData,
      y: timeLabels,
      x: freqs,
      type: 'heatmap',
      colorscale: 'Viridis',
      zmin: -50,
      zmax: 50,
      showscale: false,
    }], {
      margin: { l: 50, r: 50, t: 0, b: 50 },
      plot_bgcolor: '#1e1e1e',
      paper_bgcolor: '#1e1e1e',
      xaxis: { title: 'Freq bins', nticks: 20 },
      yaxis: { title: 'Time (slices)' },
    }, { responsive: true, staticPlot: true });

    // Helper functions
    function createVerticalLineWithBox(xData) {
      return [
        {
          type: 'line',
          x0: xData,
          x1: xData,
          y0: 0,
          y1: 1,
          yref: 'paper',
          line: {
            color: 'red',
            width: 2,
            dash: 'line'
          }
        },
        {
          type: 'rect',
          x0: xData - 100000,
          x1: xData + 100000,
          y0: -50,
          y1: 50,
          fillcolor: 'rgba(75, 75, 75, 0.2)',
          line: { width: 0 }
        }
      ];
    }

    function getXDataFromMouse(event) {
      const bb = containerFFT.getBoundingClientRect();
      const xPixel = event.clientX - bb.left;
      const xaxis = containerFFT._fullLayout.xaxis;
      return xaxis.p2l(xPixel - xaxis._offset);
    }

    function sendXDataToServer(xData) {
      fetch("/api/selected_x", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ x: xData })
      }).catch(err => console.error("Failed to send xData:", err));
    }

    function isInsideGraph(event) {
      const xaxis = containerFFT._fullLayout.xaxis;
      const yaxis = containerFFT._fullLayout.yaxis;
      const mouseX = event.clientX;
      const mouseY = event.clientY;
      const bbox = containerFFT.getBoundingClientRect();

      const plotLeft = bbox.left + xaxis._offset;
      const plotRight = plotLeft + xaxis._length;
      const plotTop = bbox.top + yaxis._offset;
      const plotBottom = plotTop + yaxis._length;

      return !(mouseX < plotLeft || mouseX > plotRight || mouseY < plotTop || mouseY > plotBottom);
    }

    // Event listeners
    const handleMouseDown = (event) => {
      if (!isInsideGraph(event)) return;
      const xData = getXDataFromMouse(event);
      const shapes = createVerticalLineWithBox(xData);
      Plotly.relayout(containerFFT, { shapes });
      sendXDataToServer(xData);
      isDragging = true;
    };

    const handleMouseMove = (event) => {
      if (!isDragging) return;
      if (!isInsideGraph(event)) {
        isDragging = false;
        return;
      }
      const xData = getXDataFromMouse(event);
      const shapes = createVerticalLineWithBox(xData);
      Plotly.relayout(containerFFT, { shapes });
      sendXDataToServer(xData);
    };

    const handleMouseUp = () => {
      isDragging = false;
    };

    const handleMouseLeave = () => {
      isDragging = false;
    };

    containerFFT.addEventListener('mousedown', handleMouseDown);
    containerFFT.addEventListener('mousemove', handleMouseMove);
    containerFFT.addEventListener('mouseup', handleMouseUp);
    containerFFT.addEventListener('mouseleave', handleMouseLeave);

    // WebSocket connection
    const ws = new WebSocket(websocketUrl);
    ws.binaryType = "arraybuffer";

    ws.onmessage = (event) => {
      if (typeof event.data === "string") {
        const json = JSON.parse(event.data);
        if (json.type === "init" || json.type === "update") {
          console.log(json.data);
        }
      } else if (event.data instanceof ArrayBuffer) {
        mags = Array.from(new Float32Array(event.data));
        fftData.push(mags);

        if (fftData.length > MAX_TIME_SLICES) {
          fftData.shift();
        }
        Plotly.update(containerFFT, { x: [freqs], y: [mags] });
        Plotly.update(containerWaterfall, { x: [freqs], z: [fftData] });
      }
    };

    ws.onopen = () => {
      ws.send(JSON.stringify({ type: "init", data: "plotly" }));
    };

    // Cleanup function
    return () => {
      containerFFT.removeEventListener('mousedown', handleMouseDown);
      containerFFT.removeEventListener('mousemove', handleMouseMove);
      containerFFT.removeEventListener('mouseup', handleMouseUp);
      containerFFT.removeEventListener('mouseleave', handleMouseLeave);
      ws.close();
    };
  }, [websocketUrl]);

  return (
    <div className="fft-plots">
      <div ref={fftContainerRef} style={{ width: '100%', height: '300px' }}></div>
      <div ref={waterfallContainerRef} style={{ width: '100%', height: '300px' }}></div>
    </div>
  );
}

export default FFTPlot;
