# Email Spam Classifier

A Python-based machine learning application that classifies emails as spam or legitimate using NLP preprocessing and scikit-learn models.

## Features

- **NLP Preprocessing**: Text cleaning, tokenization, and stemming
- **TF-IDF Vectorization**: Converts text to numerical features
- **Machine Learning Model**: Classification using scikit-learn algorithms
- **Streamlit Web Interface**: User-friendly interface for email classification
- **Pre-trained Model**: Ready-to-use trained model and vectorizer

## Project Structure

```
Email-Spam-Classifier/
├── README.md                  # Project documentation
├── app.py                     # Streamlit web application
├── data_preprocessing.ipynb   # Data preparation and model training notebook
├── model.pkl                  # Trained machine learning model
└── vectorizer.pkl            # TF-IDF vectorizer
```

## Tech Stack

- **Python 3.x**
- **scikit-learn**: Machine learning models
- **NLTK**: Natural Language Processing
- **Streamlit**: Web interface framework
- **Pandas & NumPy**: Data manipulation

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/meer140/Email-Spam-Classifier.git
   cd Email-Spam-Classifier
   ```

2. Install dependencies:
   ```bash
   pip install streamlit scikit-learn nltk pandas numpy
   ```

3. Download required NLTK data:
   ```bash
   python -m nltk.downloader stopwords punkt
   ```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

Then:
1. Enter your email text in the text area
2. Click the "Predict" button
3. View the classification result (Spam or Not Spam)

## How It Works

1. **Text Preprocessing**: 
   - Converts text to lowercase
   - Tokenizes into words
   - Removes special characters
   - Removes English stopwords
   - Applies Porter stemming

2. **Vectorization**: 
   - Transforms processed text using TF-IDF vectorizer

3. **Classification**: 
   - Makes prediction using the trained model

## Model Details

- **Training Data**: Email dataset with spam/legitimate labels
- **Algorithm**: Support for MultinomialNB or LogisticRegression
- **Features**: TF-IDF features from email text
- **Output**: Binary classification (Spam: 1, Not Spam: 0)

## Files

- `app.py`: Main Streamlit application
- `data_preprocessing.ipynb`: Jupyter notebook with data exploration and model training
- `model.pkl`: Serialized trained classification model
- `vectorizer.pkl`: Serialized TF-IDF vectorizer

## License

This project is open source and available under the MIT License.

## Author

[meer140](https://github.com/meer140)

---

**Note**: Ensure the `model.pkl` and `vectorizer.pkl` files are in the same directory as `app.py` before running the application.
