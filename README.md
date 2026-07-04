# 🛡️ Zero-Harm: AI-Powered Industrial Safety Intelligence

> **ET AI Hackathon 2026**
>
> An AI-powered Industrial Safety Intelligence platform that predicts and prevents industrial accidents using compound risk detection, real-time data fusion, and geospatial safety analytics. The system correlates gas sensors, work permits, maintenance activities, and worker locations to identify hazardous situations before incidents occur.

---

## 📖 Overview

Industrial facilities generate large volumes of data from different safety systems. However, these systems often work independently, making it difficult to identify risks that arise from multiple events occurring simultaneously.

**Zero-Harm** bridges this gap by combining data from multiple industrial sources into a unified AI-powered safety intelligence platform. Instead of monitoring individual sensors, the platform performs compound risk analysis to detect hazardous situations early and recommend preventive actions.

---

## ✨ Features

- 🔥 Compound Risk Detection Engine
- 🌡️ Live Gas Sensor Monitoring
- 📄 Digital Permit Intelligence
- 🔧 Maintenance Activity Tracking
- 👷 Worker Location Monitoring
- 🗺️ Interactive Geospatial Safety Heatmap
- 📊 Explainable AI Risk Scoring (0–100)
- 🚨 Real-Time Safety Alerts
- 🎯 Incident Simulation for Demonstration
- 📡 FastAPI REST APIs
- ⚡ Live Dashboard Updates

---

## 🏗️ System Architecture

```
IoT Gas Sensors
        │
 Permit-to-Work System
        │
 Maintenance System
        │
 Worker Tracking
        │
──────────────────────────
 Data Ingestion Layer
        │
 Compound Risk Detection Engine
        │
 Risk Scoring & Fusion
        │
 Explainable Alert Engine
        │
 Geospatial Safety Dashboard
```

---

## 🧠 How It Works

The platform continuously collects data from multiple industrial sources:

- Gas sensor readings
- Hot-work permits
- Confined-space permits
- Maintenance activities
- Worker locations

Each source is independently analyzed and then combined using a Compound Risk Detection Engine to calculate a real-time risk score. When multiple hazardous conditions occur together, the platform generates explainable alerts along with recommended preventive actions.

---

## 📊 Risk Levels

| Risk Score | Level | Recommended Action |
|------------|-------|-------------------|
| 0–19 | 🟢 Low | Continue Monitoring |
| 20–44 | 🟡 Moderate | Notify Supervisor |
| 45–69 | 🟠 High | Suspend Hazardous Activities |
| 70–100 | 🔴 Critical | Evacuate Area & Dispatch Emergency Response |

---

## 💻 Technology Stack

### Backend

- Python 3.12
- FastAPI
- Pydantic
- Uvicorn

### Frontend

- HTML5
- CSS3
- JavaScript
- SVG

### Future Integrations

- MQTT
- OPC-UA
- PostgreSQL
- PostGIS
- Knowledge Graph
- RAG + Large Language Models (LLMs)

---

## 📂 Project Structure

```
Zero-Harm/
│
├── backend/
│   ├── main.py
│   ├── risk_engine.py
│   ├── data_simulator.py
│   ├── models.py
│   ├── requirements.txt
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│
├── README.md
└── LICENSE
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/Zero-Harm.git
```

Navigate into the project

```bash
cd Zero-Harm/backend
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the server

```bash
uvicorn main:app --reload --port 8000
```

Open your browser

```
http://localhost:8000
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/dashboard` | GET | Dashboard Data |
| `/api/zones` | GET | Plant Zones |
| `/api/sensors` | GET | Sensor Readings |
| `/api/permits` | GET | Active Permits |
| `/api/maintenance` | GET | Maintenance Activities |
| `/api/workers` | GET | Worker Locations |
| `/api/alerts` | GET | Current Risk Alerts |
| `/api/simulate/incident/{zone}` | POST | Simulate Incident |
| `/api/simulate/clear/{zone}` | POST | Reset Simulation |

---

## 🎯 Demo Workflow

1. Start the FastAPI server.
2. Open the dashboard.
3. Observe all plant zones in a safe state.
4. Trigger a simulated compound incident.
5. Watch the affected zone transition from **Low → Moderate → High → Critical**.
6. View explainable AI alerts and recommended actions.
7. Reset the simulation.

---

## 🌟 Future Enhancements

- Real-Time IoT Integration
- SCADA Connectivity
- Digital Twin Integration
- AI-based Predictive Risk Analytics
- LLM-powered Safety Assistant
- Knowledge Graph-based Risk Analysis
- Automated Emergency Response
- Regulatory Compliance Intelligence

---

## 🎯 Hackathon Highlights

- ✅ Explainable AI
- ✅ Compound Risk Detection
- ✅ Multi-Source Data Fusion
- ✅ Geospatial Risk Visualization
- ✅ Real-Time Monitoring
- ✅ Scalable Microservice Architecture

---

**Building safer industries with AI.**
