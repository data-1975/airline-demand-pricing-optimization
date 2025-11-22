
---

# **easyJet Demand Forecasting & Dynamic Pricing Engine**

*A full end-to-end machine learning, optimization, and simulation pipeline for airline revenue strategy.*

---

## **📌 Overview**

This repository contains a complete implementation of an **end-to-end ML-powered dynamic pricing system** inspired by a real-world case study involving the entry of a low-cost airline (easyJet) into a competitive new market.

The project integrates:

* **Data engineering**
* **Feature store design**
* **Demand forecasting (ML)**
* **Price elasticity modeling**
* **Scenario simulation (System Dynamics + Monte Carlo)**
* **Revenue optimization using mathematical programming**
* **Real-time pricing API**

The goal is to demonstrate how airlines can use advanced analytics to:

* Predict passenger demand
* Model customer price sensitivity
* Simulate competitive scenarios
* Recommend optimal ticket prices
* Maximize revenue and load factor
* Increase strategic competitiveness

This repository is structured for clarity, reproducibility, and extensibility.

---

# **✨ Key Capabilities**

### ✔ Demand forecasting using XGBoost, LSTM, Prophet

### ✔ Price elasticity modeling

### ✔ Ensemble prediction pipeline

### ✔ System Dynamics & Monte Carlo simulation

### ✔ Revenue optimization with linear programming

### ✔ Automated pricing recommendation engine

### ✔ Production-ready FastAPI endpoint

### ✔ Modular architecture with Airflow-ready workflows

---

# **🛠 Tech Stack**

### **Languages & Frameworks**

* Python 3.9+
* FastAPI
* TensorFlow / Keras
* XGBoost
* Scikit-Learn
* Prophet / Statsmodels
* PuLP / OR-Tools
* Pandas, NumPy

### **Data & Infrastructure (optional / pluggable)**

* AWS S3 / Azure Blob / GCP Storage
* Airflow for orchestration
* MLflow for experiment tracking
* Docker (optional)

---

# **📂 Repository Structure**

```
easyjet-pricing-ml/
  ├── data_ingestion/
  │   └── load_data.py
  ├── features/
  │   └── build_features.py
  ├── models/
  │   ├── train_demand_model.py
  │   ├── train_elasticity_model.py
  │   └── forecast_demand.py
  ├── optimization/
  │   └── optimize_pricing.py
  ├── api/
  │   └── app.py
  ├── notebooks/
  │   └── exploratory_analysis.ipynb
  ├── data/
  │   ├── bronze/
  │   ├── silver/
  │   └── gold/
  ├── requirements.txt
  └── README.md
```

---

# **📊 Project Architecture**

The solution follows a **7-layer architecture**, common to modern analytics platforms.

### **1. Data Ingestion**

* Load booking history, competitor prices, calendars, and operational constraints.
* Daily ingestion handled by Airflow (or simple cron jobs).

### **2. Feature Engineering & Feature Store**

* Generate features such as:

  * Days to departure
  * Seasonality & holiday flags
  * Competitor price delta
  * Historical load factors
* Stored in `/data/gold/` for consistent model training & inference.

### **3. ML Model Training**

A hybrid modeling approach combining:

* **XGBoost** (non-linear tabular patterns)
* **LSTM** (booking curve time dependencies)
* **Prophet** (seasonality + holiday effects)
* **Quantile regression** (price elasticity)

Models are versioned and saved under `/models/`.

### **4. Forecasting Pipeline**

A daily batch job produces:

* Passenger demand forecasts
* Load factor projections
* Price sensitivity curves

Outputs are stored in `/data/gold/predicted_demand.parquet`.

### **5. Scenario Simulation**

System Dynamics + Monte Carlo allow:

* Competitor reaction scenarios
* Pricing strategy comparisons
* Occupancy risk assessment

### **6. Revenue Optimization Engine**

Using **linear programming**, maximize:

[
\text{Revenue} = \sum_i p_i \cdot q_i(p)
]

subject to:

* Aircraft capacity
* Minimum & maximum price rules
* Demand curve prediction
* Revenue management policies

### **7. Pricing API (FastAPI)**

Real-time endpoint for:

* Retrieving optimal price per flight
* Running “what-if” scenarios
* Feeding dashboards or RM systems

---

# **🚀 Quick Start**

### **1. Clone the repository**

```bash
git clone https://github.com/<your-username>/easyjet-pricing-ml.git
cd easyjet-pricing-ml
```

### **2. Install dependencies**

```bash
pip install -r requirements.txt
```

### **3. Generate features**

```bash
python features/build_features.py
```

### **4. Train demand model**

```bash
python models/train_demand_model.py
```

### **5. Run daily forecast**

```bash
python models/forecast_demand.py
```

### **6. Optimize pricing**

```bash
python optimization/optimize_pricing.py
```

### **7. Start the API**

```bash
uvicorn api.app:app --reload
```

---

# **🧠 Example: Key Scripts**

### **Demand Model (XGBoost)**

`models/train_demand_model.py` trains the primary forecasting model and reports MAE.

### **Feature Engineering**

`build_features.py` assembles temporal, competitive, and booking features.

### **Pricing Optimization**

`optimize_pricing.py` uses PuLP to solve a linear revenue maximization problem.

### **FastAPI Endpoint**

`api/app.py` exposes optimized pricing suggestions in real-time.

---

# **📈 KPIs & Metrics**

Typical evaluation metrics include:

* MAE / RMSE (demand)
* MAPE per lead-time bucket
* Elasticity stability index
* Revenue uplift
* Load factor improvement
* Price recommendation acceptance rate

---

# **📚 Use Cases**

This repository can be used for:

* Airline pricing research
* Revenue management simulations
* ML education & workshops
* Case studies for business schools
* Real-world prototyping of pricing engines

---

# **🤝 Contributing**

Contributions are welcome!

Feel free to open PRs with:

* New models
* Improved simulations
* Alternative optimization formulations
* Cloud orchestration scripts

---

# **✉ Contact**

For questions, discussions, or consulting inquiries:

**Anderson Menezes Bueno**
*Data Scientist — Advanced Analytics, Causal Inference & Optimization*

