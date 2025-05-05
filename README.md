# Emotion Detection Using SVM

This project implements a machine learning-based emotion detection system using **n-gram features** and a **Linear Support Vector Classifier (SVM)**. It classifies input text into one of seven emotion categories and displays an appropriate emoji for the detected emotion.

## 🔍 Features

- Custom n-gram feature extraction from input text.
- Multi-label emotion conversion.
- Training and evaluation using:
  - LinearSVC (Support Vector Machine).
- Display of emotion predictions using emojis.
- Input-based prediction for user-provided sentences.

## 📁 Dataset Format

The dataset is expected in a `.txt` file (`text.txt`) where each line follows the format:

```
[0 1 0 0 0 0 0] The text expressing an emotion
```

Each number in the label vector corresponds to an emotion in this order:
- `joy`, `fear`, `anger`, `sadness`, `disgust`, `shame`, `guilt`

## 📊 Emotions and Emojis

| Emotion  | Emoji |
|----------|--------|
| joy      | 😂 - Happy |
| fear     | 😱 - Fear |
| anger    | 😠 - Angry |
| sadness  | 😢 - Sad |
| disgust  | 😒 - Disgust |
| shame    | 🤭 - Shame |
| guilt    | 😳 - Guilt |

## 🛠️ Installation and Usage

1. Clone the repository:
```bash
git clone https://github.com/<your-username>/emotion-detection-svm.git
cd emotion-detection-svm
```

2. Install required libraries:
- Python 3.x
- scikit-learn
- numpy

3. Ensure your `text.txt` file is in the same directory and formatted correctly.

4. Run the script:
```bash
python emotion_detection.py
```

5. Enter a sentence when prompted and get the emotion prediction with an emoji.

## 🧠 Model Overview

- **Feature Extraction**: Custom function that generates unigram to 4-gram tokens including punctuation.
- **Vectorization**: `DictVectorizer` from `sklearn` is used to transform the text feature dictionaries into a numeric format.
- **Classifier**: `LinearSVC` is trained on the data for emotion classification.

## 📈 Sample Output

```
Enter the sentence : I can't believe this is happening!
😱 - Fear
```

## 📂 File Structure

```
emotion-detection-svm/
│
├── text.txt                # Dataset file
├── emotion_detection.py    # Main script
└── README.md               # Project description
```

## ✅ Requirements

- Python 3.x
- scikit-learn
- numpy

You can install dependencies using:
```bash
pip install scikit-learn numpy
```

## 👨‍💻 Author

- **Your Name** - [GitHub](https://github.com/your-username)
