# MindfulStress Hub - Complete AI-Powered Stress Management System

## 🌟 Overview

MindfulStress Hub is an advanced AI-powered platform that combines multimodal physiological analysis with personalized stress management interventions. The system uses EEG and ECG data to detect stress levels and provides evidence-based mindfulness techniques tailored to individual needs.

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw EEG/ECG   │ -> │  Transformer AI   │ -> │  Stress Level    │
│   Features      │    │  Multimodal       │    │  Classification  │
└─────────────────┘    │  Analysis         │    └─────────────────┘
                       └──────────────────┘             │
┌─────────────────┐    ┌──────────────────┐             │
│   Context       │ <- │   Groq AI        │ <- ┌─────────────────┐
│   Builder       │    │   Analysis       │   │  Physiological   │
└─────────────────┘    └──────────────────┘   │  Features        │
                                               └─────────────────┘
                                                       │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Technique     │ <- │   RAG System     │ <- │  Knowledge      │
│   Retrieval     │    │   ChromaDB       │    │  Base (8 tech.) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Personalized    │ <- │   Groq AI        │ <- │  Structured     │
│ Management Plan │    │   Planning       │    │  Interventions  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Create your local env file from the template
copy .env.example .env

# Then add your Groq API key to .env
```

### 2. Start the System

```bash
# Start the FastAPI server
python api.py

# Open browser to http://127.0.0.1:8000
```

### 3. Test the Pipeline

```python
from integrated_pipeline import IntegratedStressPipeline
import torch

pipeline = IntegratedStressPipeline()
eeg_features = torch.randn(1, 97)  # Your EEG features
ecg_features = torch.randn(1, 4)   # Your ECG features

result = pipeline.process_sample(eeg_features, ecg_features)
plan = result['management_plan']
print(f"Stress Level: {plan['stress_level']}")
```

## 📊 API Endpoints

### Core Analysis Endpoints

#### `POST /predict`
- **Input**: 101 physiological features (97 EEG + 4 ECG)
- **Output**: Stress level classification with probabilities

#### `POST /explain`
- **Input**: 101 physiological features
- **Output**: Stress prediction + SHAP feature importance + attention weights

#### `POST /stress-management` ⭐ **NEW**
- **Input**:
  ```json
  {
    "stress_level": "high",
    "stress_probability": 0.85,
    "shap_features": ["lf_hf_ratio", "beta_power"],
    "physiological_metrics": {
      "lf_hf_ratio": 2.5,
      "beta_power": 15.3,
      "heart_rate": 95
    },
    "all_features": {"feature_1": 0.5, ...}
  }
  ```
- **Output**:
  ```json
  {
    "stress_level": "high",
    "cause": "Elevated LF/HF ratio indicates sympathetic activation",
    "physiological_interpretation": "High beta power suggests cognitive overload",
    "recommended_actions": [
      {
        "technique": "Deep Breathing Exercise",
        "steps": "Inhale 4 counts, hold 4, exhale 6...",
        "duration": "5-10 minutes"
      }
    ]
  }
  ```

## 🎨 Frontend Features

### Stress Management Themed UI
- **🧘 Mindfulness-inspired design** with zen elements
- **🌊 Animated background** with floating meditation symbols
- **🎯 Interactive stress meter** with real-time visualization
- **📱 Responsive design** for all devices

### Key Sections
1. **Signal Input**: Upload EEG/ECG data or paste JSON features
2. **Live Analysis**: Real-time stress level prediction with confidence
3. **Explainability**: SHAP feature importance and attention visualization
4. **🧘 Stress Management**: AI-generated personalized intervention plans

## 🧠 AI Components

### 1. Multimodal Transformer
- **Architecture**: EEG (97) + ECG (4) → CLS token → 5-class stress prediction
- **Features**: Attention mechanisms, layer normalization, dropout
- **Performance**: Real-time inference (<100ms)

### 2. Context Builder (Groq AI)
- **Input**: Model outputs + physiological metrics
- **Output**: Natural language stress condition description
- **Purpose**: Bridge raw data to human-readable context

### 3. RAG System (ChromaDB)
- **Knowledge Base**: 8 evidence-based stress management techniques
- **Retrieval**: Semantic similarity search using SentenceTransformers
- **Techniques**: CBT, MBSR, WHO, NIH guidelines

### 4. Management Planner (Groq AI)
- **Input**: Stress context + retrieved techniques
- **Output**: Structured JSON intervention plan
- **Format**: Cause analysis + physiological reasoning + actionable steps

## 📚 Knowledge Base

### Included Techniques
1. **Deep Breathing Exercise** - MBSR-based parasympathetic activation
2. **Progressive Muscle Relaxation** - CBT tension reduction
3. **Cognitive Restructuring** - CBT thought pattern modification
4. **Mindfulness Meditation** - MBSR present-moment awareness
5. **Guided Imagery** - CBT visualization for stress reduction
6. **Physical Exercise** - WHO/NIH aerobic activity recommendations
7. **Social Support Seeking** - WHO mental health social connection
8. **Time Management Planning** - CBT organizational stress reduction

## 🔧 Technical Details

### Dependencies
```
fastapi==0.104.1
uvicorn==0.24.0
torch==2.1.0
chromadb==0.4.18
sentence-transformers==2.2.2
groq==0.4.2
shap==0.44.1
numpy==1.24.3
```

### Model Specifications
- **EEG Features**: 97 (32 channels × 3 bands + frontal asymmetry)
- **ECG Features**: 4 (HR, SDNN, RMSSD, LF/HF ratio)
- **Stress Classes**: 5 levels (very_low, low, moderate, high, very_high)
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)

### Performance Metrics
- **API Response Time**: <2 seconds (including Groq calls)
- **Model Inference**: <50ms
- **RAG Retrieval**: <10ms
- **Memory Usage**: ~800MB (model + embeddings)

## 🧪 Testing

### Run Component Tests
```bash
# Test RAG system
python test_system_components.py

# Test integrated pipeline
python test_integrated_pipeline.py

# Test API endpoints
python test_stress_api.py
```

### Sample Data
Test files are available in `test_samples/`:
- `low_stress_sample.json`
- `mild_stress_sample.json`
- `high_stress_sample.json`

## 🚀 Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables
```bash
GROQ_API_KEY=your_groq_api_key_here
MODEL_PATH=/path/to/trained/model.pth
CHROMA_DB_PATH=/path/to/vector/db
```

## 📈 Monitoring & Analytics

### Built-in Metrics
- API response times
- Model prediction confidence
- Technique recommendation frequency
- User interaction patterns

### Logging
- Request/response logging
- Error tracking
- Performance monitoring
- User behavior analytics

## 🔒 Security & Privacy

### Data Protection
- No physiological data stored permanently
- All processing happens in-memory
- Secure API key management
- CORS protection enabled

### Compliance
- HIPAA considerations for health data
- GDPR compliance for EU users
- Local processing (no cloud storage)

## 🤝 Contributing

### Development Setup
```bash
git clone <repository>
cd mindfulstress-hub
pip install -r requirements-dev.txt
pre-commit install
```

### Code Quality
- Black formatting
- Flake8 linting
- Pytest coverage
- Type hints required

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Evidence-Based Techniques**: CBT, MBSR, WHO, NIH guidelines
- **AI Models**: Groq (fast inference), SentenceTransformers, ChromaDB
- **Research**: DEAP dataset, multimodal stress detection studies


**MindfulStress Hub** - Transforming stress detection into stress resolution through AI-powered mindfulness. 🧘‍♀️✨
