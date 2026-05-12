# [cite_start]ArogyoAI: Clinical AI Diagnostic Platform [cite: 2]

> **Screen Smart. Act Fast. [cite_start]Stay Healthy.** [cite: 12]
> [cite_start]*An Infinity AI BuildFest — Hackathon Project Proposal for the Institute of Information Technology, Jahangirnagar University.* [cite: 3, 4]

## Executive Summary
[cite_start]ArogyoAI is an AI-powered clinical decision support platform designed to assist patients and healthcare professionals in Bangladesh[cite: 9]. [cite_start]It acts as an intelligent first-line screening layer that identifies potential conditions, quantifies confidence, and directs patients toward appropriate professional consultation[cite: 11]. [cite_start]The system integrates diagnostic modules with symptom intake, patient history management, and a real-time doctor monitoring dashboard[cite: 10]. [cite_start]We do not replace doctors; we make every doctor more powerful[cite: 11, 83].

## Problem Statement
[cite_start]Bangladesh faces a critical shortage of diagnostic infrastructure, with a national doctor-to-patient ratio of approximately 1 physician per 2,000 people[cite: 14]. [cite_start]This gap widens significantly in rural and peri-urban areas, leaving millions without timely access to basic screening[cite: 15]. Key pain points addressed include:
* [cite_start]Long waiting times and high consultation costs that deter early disease detection[cite: 17].
* Patients lacking the knowledge to self-triage and understand when testing is necessary[cite: 18].
* [cite_start]Diagnostic images being reviewed days after acquisition due to the unavailability of specialists[cite: 19].
* [cite_start]The lack of a unified system to link patient symptoms, test results, and follow-up history[cite: 20].

## Core Features

### Symptom Questionnaire & Flow
* Patients complete a structured symptom intake form before any diagnostic test is initiated[cite: 27].
* [cite_start]The form adapts based on answers, such as presenting follow-up questions about duration and radiation if chest pain is reported[cite: 28].
* [cite_start]It supports both English and Bangla input via Groq API language processing[cite: 29].
* The system outputs a recommended test or set of tests based on symptom clustering[cite: 30].

### AI Diagnostic Modules
The platform features four AI-powered modules that accept medical files and return class probabilities and confidence scores[cite: 34]:
* [cite_start]**Chest X-Ray**: Uses an ensemble of DenseNet121, ResNet50, and ViT-Base to detect conditions like Cardiomegaly, Edema, Consolidation, Pneumonia, and Atelectasis from JPEG/PNG inputs[cite: 35].
* [cite_start]**ECG**: Utilizes a 1D ResNet + Transformer to classify 360 Hz CSV signals into categories like Normal, AF, ST-Elevation, and Bundle Branch Block[cite: 35].
* **CT Scan**: A planned 3D CNN for nodule and mass detection from DICOM or PNG slices[cite: 35].
* [cite_start]**Skin Disease**: Leverages EfficientNet-B3 to identify Melanoma, Eczema, Psoriasis, Acne, and more[cite: 35].

### Risk Scoring & Alerts
* [cite_start]Each AI prediction includes a confidence percentage, and results below 60% confidence are explicitly flagged as inconclusive[cite: 37].
* All test results and symptom scores are aggregated into a single Overall Risk Score (0–100) per patient session[cite: 40].
* [cite_start]High-risk scores (61–100) trigger an immediate emergency alert pushed to the doctor, and the patient is advised to seek urgent care[cite: 41, 61].

### Visual Explainability & Recommendations
* [cite_start]For X-Ray and Skin Disease modules, the system generates a Gradient-weighted Class Activation Map (Grad-CAM) overlay[cite: 50].
* This heatmap highlights the specific regions of the image that most influenced the prediction, increasing physician trust and transparency[cite: 51, 52].
* [cite_start]Groq API (LLaMA 3) uses the patient's full history to generate personalized, evidence-based recommendations, including dietary adjustments and physical activity guidelines[cite: 43, 44, 45].
* [cite_start]No pharmaceutical drug names or dosages are ever suggested by the AI[cite: 48].

## Doctor Dashboard
[cite_start]A dedicated interface provides oversight for registered physicians, offering tools for review and direct communication[cite: 63, 64].
* [cite_start]**Patient List**: Ranks all patients by Risk Score and highlights critical patients in orange[cite: 65].
* **Individual Reports**: Displays the full AI result, confidence scores, Grad-CAM images, and symptom history for each patient[cite: 65].
* [cite_start]**Doctor Messaging & Scheduling**: Features secure in-app messaging for follow-up instructions and allows doctors to schedule sessions directly from the patient record[cite: 65].
* [cite_start]**AI Accuracy Review**: Allows doctors to mark results as confirmed or incorrect to create a feedback loop for model improvement[cite: 65, 78].

## System Architecture
* **Frontend**: Built with React.js, HTML, and Tailwind CSS[cite: 67]. Deployed via Netlify[cite: 67].
* [cite_start]**Backend API**: Powered by FastAPI (Python) to manage REST endpoints, auth, inference calls, and alert logic[cite: 67]. [cite_start]Hosted on Render[cite: 67].
* **Machine Learning**: Built on PyTorch using models like DenseNet121, ResNet50, ViT-Base, 1D ResNet, and EfficientNet[cite: 67].
* [cite_start]**AI Language Integration**: Groq API (LLaMA 3) is used for symptom processing, Bangla support, and recommendation generation[cite: 67].
* [cite_start]**Database**: MongoDB Atlas handles patient profiles, test history, messages, and doctor records[cite: 67].

## Ethical Commitments & Responsible AI
* ArogyoAI is strictly a decision-support tool, not a diagnostic authority; all results require physician validation[cite: 73].
* [cite_start]Confidence scores and inconclusiveness flags are always shown to prevent blind reliance on AI output[cite: 75].
* [cite_start]Patient data is securely stored with access controls, and doctor-patient communication is kept private[cite: 76].
* Grad-CAM explainability ensures the AI cannot function as a black box[cite: 77].
