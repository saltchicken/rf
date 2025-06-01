import React, { useState, useMemo } from 'react';
import debounce from 'lodash/debounce';

const CenterFreqControl = () => {
  const [number, setNumber] = useState('');

  const sendNumber = async (value) => {
    try {
      const res = await fetch("http://localhost:5000/api/set-setting", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ setting: "center_freq", value: parseFloat(value) }),
      });

      const data = await res.json();
      console.log("Response from FastAPI:", data.message);
    } catch (err) {
      console.error("Error calling FastAPI:", err);
    }
  };

  const debouncedSendNumber = useMemo(
    () => debounce(sendNumber, 500),
    [] // only create once
  );

  const handleChange = (e) => {
    const value = e.target.value;
    setNumber(value);

    if (value !== '') {
      const parsed = parseFloat(value);
      if (!isNaN(parsed)) {
        debouncedSendNumber(parsed);
      }
    }
  };

  return (
    <div style={{ padding: '1em' }}>
      <label
        htmlFor="center-freq-input"
        style={{ display: 'block', marginBottom: '0.5em', fontWeight: 'bold' }}
      >
        Center Freq
      </label>
      <input
        id="center-freq-input"
        type="number"
        value={number}
        onChange={handleChange}
        placeholder="Enter a number"
        style={{ padding: '0.5em', fontSize: '1rem', width: '100px' }}
      />
    </div>
  );
};

export default CenterFreqControl;
