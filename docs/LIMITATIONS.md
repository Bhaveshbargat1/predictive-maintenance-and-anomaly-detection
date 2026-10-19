# System Limitations & Future Engineering Enhancements

## 1. Current Limitations

### 1.1 Synthetic Dataset Nature
- The AI4I 2020 dataset, while calibrated to realistic mechatronic parameters and verified by academic literature (Matzka 2020), is generated from a digital twin model.
- Real shop-floor telemetry contains complex electromagnetic interference (EMI), intermittent sensor dropouts, packet jitter, and non-stationary ambient shop-floor temperature shifts between Indian summer and winter seasons.

### 1.2 Static Tabular Representation vs High-Frequency Waveforms
- Current data aggregates readings into static tabular rows rather than continuous multi-kilohertz raw vibration accelerometer waveforms (FFT spectra) or acoustic emission signals.
- Bearing micro-crack flaking is typically detectable weeks earlier in raw vibration frequency spectra before torque or tool wear reflects degradation.

### 1.3 Stationary Tool Wear Assumptions
- The model treats tool wear accumulation linearly, whereas real tungsten carbide inserts exhibit three distinct wear phases:
  1. Initial break-in (rapid steep wear)
  2. Steady-state uniform wear
  3. Accelerated catastrophic thermal collapse (exponential wear rate)

### 1.4 Centralized Inference vs Ultra-Low-Latency Edge Deployment
- In high-speed milling operations (10,000+ RPM), a spindle collision or catastrophic tool fracture occurs within 10 to 50 milliseconds.
- A web application dashboard (Streamlit) operates at 100-500 ms latency, making it suitable for supervisory predictive alerts, but requiring hard real-time PLC/FPGA interrupts for instantaneous emergency tripping.

## 2. Roadmap & Production Hardening
1. **Edge Deployment**: Export trained Random Forest and Isolation Forest to ONNX runtime or C++ micro-models on edge gateways (Siemens SIMATIC IOT2050 or Advantech IPC).
2. **Vibration Waveform Integration**: Ingest 3-axis accelerometer feeds via MQTT/OPC-UA and compute rolling RMS, Kurtosis, Crest Factor, and spectral peak energies.
3. **Continuous Retraining Pipeline**: Implement automated MLOps pipelines (MLflow + GitHub Actions) to detect concept drift and retrain models quarterly.
4. **Integration with Plant ERP/CMMS**: Automatically dispatch work orders to SAP PM or IBM Maximo when failure risk exceeds critical thresholds.
