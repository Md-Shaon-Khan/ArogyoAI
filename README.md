# ArogyoAI: Clinical AI Diagnostic Platform

## Screen Smart. Act Fast. Stay Healthy.

*A Project Proposal for the Infinity AI BuildFest Hackathon, Institute of Information Technology (IIT), Jahangirnagar University*

---

# Executive Summary

ArogyoAI is an AI-powered Clinical Decision Support Platform designed to assist patients and healthcare professionals in Bangladesh through intelligent preliminary screening and risk assessment.

The platform functions as a first-line diagnostic assistance system that analyzes patient symptoms, medical reports, and diagnostic images to identify potential health conditions and recommend appropriate medical consultation.

ArogyoAI integrates multiple AI diagnostic modules with adaptive symptom intake, patient history management, and a real-time doctor monitoring dashboard. The platform is designed to enhance healthcare efficiency and support physicians in making faster and more informed clinical decisions.

The system does not replace doctors. Instead, it strengthens healthcare delivery through AI-assisted screening and explainable diagnostic support.

---

# Problem Statement

Bangladesh faces major healthcare accessibility challenges due to a shortage of doctors and diagnostic infrastructure, especially in rural and peri-urban regions.

Several key problems motivate the development of ArogyoAI:

- Long waiting times and high consultation costs discourage early disease detection.
- Many patients cannot determine when professional testing or urgent care is necessary.
- Diagnostic images and reports are often reviewed late because of specialist shortages.
- Healthcare systems frequently lack centralized patient monitoring and historical tracking.
- Delayed diagnosis increases treatment complexity and healthcare burden.

ArogyoAI aims to reduce these gaps by providing intelligent screening support, risk prioritization, and physician-assisted decision support.

---

# Core Features

## 1. Adaptive Symptom Questionnaire System

The platform begins with a structured symptom intake process before diagnostic analysis.

### Features

- Dynamic questionnaires adapt according to patient responses.
- Follow-up questions are generated based on symptom severity, duration, and related conditions.
- Supports both Bangla and English interaction through Groq API language processing.
- Recommends appropriate diagnostic tests using symptom clustering and AI reasoning.

---

## 2. AI Diagnostic Modules

ArogyoAI integrates multiple AI-powered diagnostic systems capable of analyzing medical data and generating confidence-based predictions.

### Chest X-Ray Analysis

Uses an ensemble architecture consisting of:

- DenseNet121
- ResNet50
- Vision Transformer (ViT-Base)

Detects conditions such as:

- Pneumonia
- Cardiomegaly
- Pulmonary Edema
- Consolidation
- Atelectasis

Supported formats:

- JPEG
- PNG

---

### ECG Signal Analysis

Uses a hybrid architecture combining:

- 1D ResNet
- Transformer Networks

Classifies ECG conditions including:

- Normal Rhythm
- Atrial Fibrillation (AF)
- ST-Elevation
- Bundle Branch Block

Input format:

- 360 Hz CSV signals

---

### CT Scan Analysis (Planned Module)

Future implementation using 3D CNN architecture for:

- Nodule Detection
- Mass Detection
- Tissue Abnormality Localization

Supported formats:

- DICOM
- PNG Slices

---

### Skin Disease Detection

Built using EfficientNet-B3 architecture.

Detects conditions including:

- Melanoma
- Eczema
- Psoriasis
- Acne
- Other common skin diseases

---

# Risk Scoring and Emergency Alert System

Each AI-generated prediction includes:

- Probability distributions
- Confidence percentages
- Inconclusive result detection

Predictions below 60% confidence are automatically flagged as inconclusive.

The platform combines:

- Symptom severity
- Diagnostic results
- Patient history
- AI confidence scores

into a unified Overall Risk Score (0-100).

## Risk Classification

| Risk Score | Classification |
|------------|----------------|
| 0-30       | Low Risk       |
| 31-60      | Moderate Risk  |
| 61-100     | High Risk      |

### High-Risk Alert Actions

- Immediate emergency alert sent to doctors
- Priority patient highlighting
- Urgent consultation recommendation

---

# Visual Explainability and Transparency

To improve physician trust and system transparency, ArogyoAI incorporates Explainable AI techniques.

## Grad-CAM Visualization

For X-Ray and Skin Disease modules, the system generates:

- Gradient-weighted Class Activation Maps (Grad-CAM)

These heatmaps highlight image regions that most influenced the AI prediction.

### Benefits

- Improves interpretability
- Reduces black-box concerns
- Supports physician validation
- Increases trust in AI-assisted diagnosis

---

# Personalized Health Recommendations

Using Groq API with LLaMA 3 integration, the platform generates evidence-based recommendations based on:

- Patient history
- Symptoms
- Risk scores
- Diagnostic outcomes

### Recommendation Types

- Lifestyle guidance
- Dietary suggestions
- Physical activity recommendations
- Follow-up consultation advice

### Ethical Limitation

The AI system does not:

- Prescribe medication
- Suggest pharmaceutical drug names
- Recommend dosages

---

# Doctor Dashboard

A dedicated physician dashboard provides monitoring and clinical review functionality.

## Features

### Patient Prioritization

- Patients ranked according to risk score
- Critical patients visually highlighted

### Individual Patient Reports

Doctors can access:

- AI predictions
- Confidence scores
- Grad-CAM visualizations
- Symptom history
- Uploaded medical records

### Secure Communication

- In-app doctor-patient messaging
- Follow-up scheduling
- Secure consultation support

### AI Feedback Loop

Doctors can:

- Confirm predictions
- Mark incorrect outputs
- Provide feedback for future model improvement

---

# System Architecture

## Frontend

Technologies used:

- React.js
- HTML
- Tailwind CSS

Deployment:

- Netlify

---

## Backend

Built using:

- FastAPI (Python)

Responsibilities:

- REST API management
- Authentication
- AI inference handling
- Risk alert processing

Deployment:

- Render

---

## Machine Learning Stack

Framework:

- PyTorch

Models used:

- DenseNet121
- ResNet50
- Vision Transformer (ViT-Base)
- EfficientNet-B3
- 1D ResNet
- Transformer Networks

---

## AI Language Integration

Groq API with LLaMA 3 supports:

- Symptom understanding
- Bangla language processing
- Recommendation generation

---

## Database

MongoDB Atlas is used for:

- Patient profiles
- Medical history
- Diagnostic records
- Messaging data
- Doctor management

---

# Ethical Commitments and Responsible AI

ArogyoAI is strictly a Clinical Decision Support System and not an autonomous diagnostic authority.

All outputs require physician validation before medical action is taken.

## Ethical Principles

- Confidence scores are always displayed
- Inconclusive predictions are clearly identified
- Patient data is securely stored
- Access control mechanisms protect privacy
- Doctor-patient communication remains confidential
- Explainable AI reduces black-box dependency
- Physicians remain the final decision-makers

---

# Conclusion

ArogyoAI aims to improve healthcare accessibility in Bangladesh through AI-assisted screening, explainable diagnostics, multilingual interaction, and risk-based prioritization.

The platform is designed to support healthcare professionals by enabling faster screening, earlier detection, and more informed decision-making.

Rather than replacing doctors, ArogyoAI strengthens healthcare systems through responsible and transparent AI integration.
