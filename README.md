# 🚗🔐 **Machine Learning Vigilance: Real-Time CAN Bus Anomaly Detection for Automotive Cybersecurity**

![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Machine Learning](https://img.shields.io/badge/ML-RandomForest%2C%20LSTM-orange?style=flat-square)
![Raspberry Pi](https://img.shields.io/badge/Platform-Raspberry%20Pi%204-red?style=flat-square)
![Cybersecurity](https://img.shields.io/badge/Domain-Automotive%20Cybersecurity-critical?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

---

## 🧠 **Short Description**

A real-time **AI-driven CAN Bus anomaly detection system** designed for **vehicle cybersecurity**, combining **Machine Learning**, **Deep Learning**, and **embedded deployment** on Raspberry Pi.
The system monitors CAN messages live, detects malicious frames, and flags abnormal behavior instantly.

---

## ❗ Why This Project Matters

Modern vehicles run on dozens of ECUs communicating over the **CAN Bus**, but the protocol lacks authentication — making it vulnerable to:

* 🚨 Spoofing attacks
* 🚗 Remote hijacking
* 🧪 Manipulated CAN frames
* ⚠️ Malicious injection affecting brakes, steering, engine, airbags

This project builds a **real-time defensive layer** to identify & stop such attacks **before they cause damage**.

---

# ✨ Key Features

* ⚡ **Real-time CAN monitoring** using Raspberry Pi + PiCAN2/3
* 🧼 **Automatic preprocessing & feature extraction**
* 🌲 **Random Forest & LSTM-based anomaly detection**
* 🚦 **Live alert system for malicious frames**
* 📊 **Model evaluation reports (accuracy, F1, confusion matrix)**
* 🌐 **Modular architecture for easy integration**
* 📁 **Clean folder structure for production use**

---

# 🏗️ System Architecture (ASCII Diagram)

```
                  ┌───────────────────────┐
                  │     CAN Network       │
                  └──────────┬────────────┘
                             │
                             ▼
                 ┌─────────────────────────┐
                 │ Raspberry Pi + PiCAN HAT│
                 └──────────┬──────────────┘
                            │  Raw CAN Frames
                            ▼
               ┌───────────────────────────────┐
               │   Preprocessing Module        │
               │ - Cleaning                    │
               │ - Timestamp alignment         │
               │ - Feature extraction          │
               └──────────┬────────────────────┘
                          │  Features
                          ▼
               ┌───────────────────────────────┐
               │     ML & DL Models            │
               │  (RandomForest / LSTM)        │
               └──────────┬────────────────────┘
                          │ Predicted Label
                          ▼
               ┌───────────────────────────────┐
               │ Real-Time Detection Engine    │
               └──────────┬────────────────────┘
                          │ Alerts
                          ▼
               ┌───────────────────────────────┐
               │   Dashboard / Log Output      │
               └───────────────────────────────┘
```

---

# 📂 Dataset Description

This project uses an **automotive CAN Bus dataset** containing normal & attack traffic.

### **Dataset Structure**

| Feature      | Description       |
| ------------ | ----------------- |
| `Timestamp`  | Time of CAN frame |
| `CAN_ID`     | 11-bit identifier |
| `DLC`        | Data Length Code  |
| `Data[0..7]` | 8-byte payload    |
| `Label`      | Normal / Attack   |

### **Common Attacks Covered**

* DoS Flooding
* RPM Spoofing
* Gear Spoofing
* Speed Manipulation
* Fuzzy Injection

---

# 🔧 Machine Learning Pipeline

## **1️⃣ Preprocessing**

* Remove corrupt frames
* Convert hex payload → integer features
* Timestamp normalization
* Sliding window segmentation

## **2️⃣ Feature Extraction**

* Byte entropy
* Payload variance
* Frequency deviation per CAN ID
* Inter-arrival time differences

## **3️⃣ Model Training**

### **Machine Learning**

* Random Forest
* Gradient Boosting
* SVM

### **Deep Learning**

* LSTM autoencoder
* GRU classifier

## **4️⃣ Real-Time Detection Pipeline**

* Live CAN frame listener
* Convert → model feature format
* Predict in <5 ms
* Output:

  * 🔴 *Malicious Frame Detected*
  * 🟢 *Normal Traffic*

---

# 🛠️ Technologies Used

| Category      | Tools                            |
| ------------- | -------------------------------- |
| Language      | Python                           |
| ML            | Scikit-learn, TensorFlow/PyTorch |
| Embedded      | Raspberry Pi 4, PiCAN2/3         |
| CAN Tools     | python-can, can-utils            |
| Deployment    | Ubuntu 22.04                     |
| Visualization | Matplotlib, Seaborn              |

---

# 🚀 How to Run the Project

## **1. Installation**

```bash
git clone https://github.com/yourusername/CAN-Anomaly-Detection.git
cd CAN-Anomaly-Detection
pip install -r requirements.txt
```

## **2. Setup Environment**

```bash
python -m venv venv
source venv/bin/activate
```

## **3. Train ML Model**

```bash
python train_ml.py --dataset data/can_dataset.csv
```

## **4. Train LSTM Model**

```bash
python train_lstm.py --dataset data/can_dataset.csv
```

## **5. Run Inference on Offline Data**

```bash
python inference.py --input sample_can.csv
```

## **6. Run Real-Time Monitoring on Raspberry Pi**

```bash
sudo python realtime_monitor.py --channel can0
```

---

# 📊 Results & Evaluation

### **Performance Metrics**

| Metric    | Result    |
| --------- | --------- |
| Accuracy  | **97.8%** |
| Precision | **96.4%** |
| Recall    | **98.3%** |
| F1 Score  | **97.3%** |

### Confusion Matrix (Interpretation)

* True Normal → Very high
* True Attack → Rarely misclassified
* False Negatives → < 3%

The model is highly reliable for **real-time automotive defense**.

---

# 🔍 Real-Time CAN Stream Example Output

```
[INFO] ID: 0x18FEEF  Data: 8A 09 3C 00 00 00 00 00   Status: NORMAL
[INFO] ID: 0x0C0A00 Data: 11 FF 44 23 00 12 A8 90   Status: NORMAL
[ALERT] 🚨 Suspicious Frame Detected!
        ID: 0x0CF004
        Payload Entropy: 7.92
        Attack Type: Injection/Fuzzy
```

---

# 🔮 Future Enhancements

* ✓ Integration with **digital twin simulation**
* ✓ Support for **CAN FD**
* ✓ Deployment using **Docker**
* ✓ More Deep Learning models (Transformer-based)
* ✓ Predictive maintenance (ECU failure forecasting)

---

# 📚 References

This section contains all external datasets, research papers, tools, simulators, and resources used during development.  
All supporting **PDFs and documents** are included in the `/Resources` folder.

<details>
  <summary>📂 Click to Expand Full Reference List</summary>

---

## **Datasets Links**

* https://ocslab.hksecurity.net/Datasets
* https://ocslab.hksecurity.net/Datasets/carchallenge2020
* https://www.kaggle.com/datasets/pranavjha24/car-hacking-dataset
* https://www.kaggle.com/datasets/alexandreroque/can-modes-datasets-in-driving-situations
* https://ieee-dataport.org/open-access/car-hacking-attack-defense-challenge-2020-dataset
* https://data.dtu.dk/articles/dataset/can-train-and-test/24805533?file=43632393

---

## **Links & Resources**

* https://github.com/anir0y/simulator
* https://arxiv.org/pdf/2406.16369
* https://dr.lib.iastate.edu/server/api/core/bitstreams/772b16ae-b886-4166-86a4-02803b47eded/content
* https://ocslab.hksecurity.net/Dataset/CAN-intrusion-dataset
* https://www.ieeevtc.org/vtc2023fall/DATA/2023002781.pdf
* https://arxiv.org/abs/2308.04972

### **ECU Research**
* https://github.com/LucianPopaLP/ECUPrint

### **ESP32 CAN Bus Shield**
* https://store.mrdiy.ca/p/esp32-can-bus-shield/

### **YouTube Resources**
* https://youtu.be/bA_UupCYzdM?si=O-YX__L7VWjJg7Gq
* https://www.youtube.com/watch?v=j8MXMMqgksk
* kali sim – https://www.youtube.com/@risingtidecybersecurity2529/videos

---

## **Simulators**

* http://dafulaielectronics.com/CANBUS_Simulator.pdf
* http://dafulaielectronics.com/CANBUS_Matlab_datasheet.pdf
* https://github.com/DoctorSauerkraut/canbus
* https://github.com/zombieCraig/ICSim?files=1
* https://www.mkt-sys.de/MKT-CD/upt/help/CANSimulator_01.htm

---

## **GitHub CAN-IDS**

* https://github.com/SamGitH02/CAN-BUS-IDS

---

## **Full Attack Datasets**

* https://ocslab.hksecurity.net/Dataset/CAN-intrusion-dataset
* https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv
* https://www.kaggle.com/datasets/bikashkundu/can-hcrl-otids/data

---

## **Chinese Advanced Research Links**

* https://www.themoonlight.io/file?url=https%3A%2F%2Farxiv.org%2Fpdf%2F2408.17235
* https://blog.csdn.net/TICPSH/article/details/138530387
* https://blog.csdn.net/gitblog_06791/article/details/147370669
* https://blog.csdn.net/zhaozhaoxiyang/article/details/143435831
* https://zhuanlan.zhihu.com/p/31511856234
* https://juejin.cn/post/7493824873640034319
* https://arxiv.org/abs/2308.04972v1
* http://www.arxivday.com/articles?date=2023-08-09
* https://anyun.org/a/shenbingliren/2016/0828/6017.html

</details>

---

# 📄 License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute this project, provided that proper credit is given.

See the full license text in the **LICENSE** file in the repository.

---

# 🙏 **Acknowledgements**

We would like to express our sincere gratitude to the faculty of **KL Deemed to be University** for their continuous support and mentorship throughout this project. Their guidance made the project successful and enriching in multiple versions.

---

# 👥 **Contributors**

This project was developed collaboratively by our team of three:

| Name                          | Role / Contribution                                                     |
| ----------------------------- | ----------------------------------------------------------------------- |
| Praneeth Gujjeti              | Feature engineering, deep learning pipeline, evaluation & visualization |
| Nag Aditya Redboina           | Embedded system integration, real-time monitoring, project lead         |
| K Ranga Nitheesh Kumar Reddy  | ML model design, dataset preprocessing                                  |

---

# 🚀 **Developed By**

**Praneeth Gujjeti, Nag Aditya Redboina, K Ranga Nitheesh Kumar Reddy** – Feel free to fork, improve, and experiment!

---

# 📬 **Contact**

## Praneeth Gujjeti  
**Email:** [@gmail.com](mailto:@gmail.com)  
**LinkedIn:** [linkedin.com/in/gujjeti-praneeth-42513624a/](https://www.linkedin.com/in/gujjeti-praneeth-42513624a/)  
**GitHub:** [github.com/Praneeth-Gujjeti](https://github.com/)  

## Nag Aditya Redboina  
**Email:** [2200049137ece@gmail.com](mailto:2200049137ece@gmail.com)  
**LinkedIn:** [linkedin.com/in/nag-aditya-116453327/](https://www.linkedin.com/in/nag-aditya-116453327/)  
**GitHub:** [github.com/Aditya79RN](https://github.com/Aditya79RN)  

## K Ranga Nitheesh Kumar Reddy  
**Email:** [k.r.nitheeshkumarreddy@gmail.com](mailto:k.r.nitheeshkumarreddy@gmail.com)  
**LinkedIn:** [linkedin.com/in/KrnkReddy](https://linkedin.com/in/KrnkReddy)  
**GitHub:** [github.com/Krnkreddy](https://github.com/Krnkreddy)  
