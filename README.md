# Audio Spectrogram Classifier

**Dataset used:** https://www.kaggle.com/datasets/murtadhanajim/gender-recognition-by-voiceoriginal (organized in folders per class)

## Setup

- Clone the repository
- Install dependencies:

```bash
pip install tensorflow matplotlib numpy
```

## Training Summary

- **Total parameters:** 7,392,005 (28.20 MB)  
- **Trainable parameters:** 7,392,002 (28.20 MB)  
- **Non-trainable parameters:** 3 (16.00 B)

### Training Results

The model was trained for up to 10 epochs with early stopping. Key metrics from training:

| Epoch | Accuracy | Loss   | Val Accuracy | Val Loss |
|-------|---------|--------|--------------|----------|
| 1     | 0.9452  | 0.1413 | 0.9944       | 0.0200   |
| 2     | 0.9948  | 0.0159 | 0.9962       | 0.0140   |
| 3     | 0.9978  | 0.0064 | 0.9962       | 0.0147   |
| 4     | 0.9993  | 0.0025 | 0.9850       | 0.0513   |

- **Early stopping** was triggered at epoch 4, restoring the model to the best weights from epoch 2.  
- **Final evaluation on test set:**  
  - Accuracy: 0.9963  
  - Loss: 0.0102

## Images

### Example Waveform Analysis
![Waveform Analysis](images/waveform-analysis.png)

### Waveform vs Spectrogram Comparison
![Waveform-Spectrogram Comparison 1](images/waveform-spectrogram-comparison-1.png)

![Waveform-Spectrogram Comparison 2](images/waveform-spectrogram-comparison-2.png)

### Training history
![Model Architecture](images/training-history.png)

### Model Architecture
![Model Architecture](images/model.png)