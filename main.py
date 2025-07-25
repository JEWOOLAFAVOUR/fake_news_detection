import pickle
import numpy as np
import re
import pandas as pd
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from collections import Counter
import uvicorn

# Define the EnhancedNaiveBayes class
class EnhancedNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_probs = {}
        self.feature_means = {}
        self.feature_stds = {}
        
    def fit(self, X, y):
        classes = np.unique(y)
        n_samples, n_features = X.shape
        
        for c in classes:
            class_mask = (y == c)
            self.class_probs[c] = np.sum(class_mask) / n_samples
            class_features = X[class_mask]
            self.feature_means[c] = np.mean(class_features, axis=0)
            self.feature_stds[c] = np.std(class_features, axis=0) + 1e-6
    
    def _gaussian_probability(self, x, mean, std):
        exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        return exponent / (np.sqrt(2 * np.pi) * std)
    
    def predict_proba(self, X):
        probabilities = []
        
        for sample in X:
            class_probs = {}
            
            for c in self.class_probs:
                prob = np.log(self.class_probs[c])
                for i, feature_val in enumerate(sample):
                    feature_prob = self._gaussian_probability(
                        feature_val, 
                        self.feature_means[c][i], 
                        self.feature_stds[c][i]
                    )
                    prob += np.log(feature_prob + 1e-10)
                class_probs[c] = prob
            
            max_log_prob = max(class_probs.values())
            exp_probs = {c: np.exp(prob - max_log_prob) for c, prob in class_probs.items()}
            total_prob = sum(exp_probs.values())
            normalized_probs = {c: prob / total_prob for c, prob in exp_probs.items()}
            probabilities.append([normalized_probs.get(0, 0), normalized_probs.get(1, 0)])
        
        return np.array(probabilities)
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return (probas[:, 1] > 0.5).astype(int)

app = FastAPI()

# Load the model and components
with open('fake_news_model.pkl', 'rb') as f:
    model_components = pickle.load(f)

TRAINED_MODEL = model_components['model']
VOCAB = model_components['vocab']
POS_COLUMNS = pd.Index(model_components['pos_columns'])
NER_COLUMNS = pd.Index(model_components['ner_columns'])
LINGUISTIC_COLUMNS = pd.Index(model_components['linguistic_columns'])
LINGUISTIC_STATS = model_components['linguistic_stats']

# Preprocessing and feature extraction functions
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s.,!?;:]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_pos_features_simple(text):
    if not text or pd.isna(text):
        return {}
    words = text.split()
    total_words = len(words)
    if total_words == 0:
        return {}
    features = {}
    past_verbs = len([w for w in words if w.endswith('ed')])
    ing_verbs = len([w for w in words if w.endswith('ing')])
    plural_nouns = len([w for w in words if w.endswith('s') and len(w) > 3])
    adverbs = len([w for w in words if w.endswith('ly')])
    articles = len([w for w in words if w in ['the', 'a', 'an', 'this', 'that', 'these', 'those']])
    prepositions = len([w for w in words if w in ['in', 'on', 'at', 'by', 'for', 'with', 'about', 'to', 'from']])
    pronouns = len([w for w in words if w in ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them']])
    features['pos_past_verbs'] = past_verbs / total_words
    features['pos_ing_verbs'] = ing_verbs / total_words
    features['pos_plural_nouns'] = plural_nouns / total_words
    features['pos_adverbs'] = adverbs / total_words
    features['pos_articles'] = articles / total_words
    features['pos_prepositions'] = prepositions / total_words
    features['pos_pronouns'] = pronouns / total_words
    return features

def extract_ner_features_simple(text):
    if not text or pd.isna(text):
        return {}
    words = text.split()
    total_words = len(words)
    if total_words == 0:
        return {}
    features = {}
    capitalized = len([w for w in words if w[0].isupper() and len(w) > 1])
    common_first_names = ['john', 'donald', 'hillary', 'barack', 'joe', 'mike', 'sarah', 'nancy']
    person_names = len([w for w in words if w.lower() in common_first_names])
    location_words = ['america', 'usa', 'washington', 'california', 'texas', 'florida', 'york', 'city', 'state', 'country']
    locations = len([w for w in words if w.lower() in location_words])
    org_words = ['government', 'congress', 'senate', 'house', 'department', 'agency', 'company', 'corporation']
    organizations = len([w for w in words if w.lower() in org_words])
    political_words = ['republican', 'democrat', 'party', 'election', 'vote', 'campaign', 'president', 'senator']
    political_entities = len([w for w in words if w.lower() in political_words])
    features['ner_capitalized'] = capitalized / total_words
    features['ner_person_names'] = person_names / total_words
    features['ner_locations'] = locations / total_words
    features['ner_organizations'] = organizations / total_words
    features['ner_political'] = political_entities / total_words
    return features

def extract_linguistic_features(text):
    if not text or pd.isna(text):
        return {}
    features = {}
    features['text_length'] = len(text)
    words = text.split()
    features['word_count'] = len(words)
    features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
    features['sentence_count'] = len(re.split(r'[.!?]+', text))
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['comma_count'] = text.count(',')
    features['period_count'] = text.count('.')
    features['quote_count'] = text.count('"') + text.count("'")
    features['capital_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    sensational_words = ['shocking', 'breaking', 'exclusive', 'urgent', 'secret', 'revealed', 'exposed', 'scandal']
    features['sensational_words'] = len([w for w in words if w.lower() in sensational_words]) / len(words) if words else 0
    features['emotional_words'] = len([w for w in words if w.lower() in ['amazing', 'terrible', 'incredible', 'unbelievable', 'outrageous', 'disgusting']]) / len(words) if words else 0
    return features

def text_to_tfidf(texts, vocab):
    vectors = []
    N = len(texts)
    df = {}
    for text in texts:
        words = set(text.split())
        for word in words:
            if word in vocab:
                df[word] = df.get(word, 0) + 1
    for text in texts:
        words = text.split()
        vector = [0] * len(vocab)
        word_count = Counter(words)
        total_words = len(words)
        for word, idx in vocab.items():
            if word in word_count:
                tf = word_count[word] / total_words
                idf = np.log(N / (df.get(word, 1) + 1))
                vector[idx] = tf * idf
        vectors.append(vector)
    return np.array(vectors)

# Modified test_fake_news function for API
def test_fake_news(article_text):
    try:
        processed_text = preprocess_text(article_text)
        if not processed_text:
            return {"error": "Invalid or empty text provided"}
        
        pos_features = extract_pos_features_simple(processed_text)
        ner_features = extract_ner_features_simple(processed_text)
        ling_features = extract_linguistic_features(processed_text)
        
        for key, value in ling_features.items():
            if key in LINGUISTIC_STATS:
                mean = LINGUISTIC_STATS[key]['mean']
                std = LINGUISTIC_STATS[key]['std']
                if std > 0:
                    ling_features[key] = (value - mean) / std
        
        tfidf_vector = text_to_tfidf([processed_text], VOCAB)[0]
        
        feature_vector = []
        for col in POS_COLUMNS:
            feature_key = col.replace('pos_', '')
            feature_vector.append(pos_features.get(feature_key, 0))
        for col in NER_COLUMNS:
            feature_key = col.replace('ner_', '')
            feature_vector.append(ner_features.get(feature_key, 0))
        for col in LINGUISTIC_COLUMNS:
            feature_vector.append(ling_features.get(col, 0))
        feature_vector.extend(tfidf_vector)
        
        prediction_prob = TRAINED_MODEL.predict_proba([feature_vector])[0]
        prediction_class = TRAINED_MODEL.predict([feature_vector])[0]
        
        fake_confidence = prediction_prob[0]
        true_confidence = prediction_prob[1]
        
        final_prediction = "FAKE" if prediction_class == 0 else "TRUE"
        confidence = fake_confidence if prediction_class == 0 else true_confidence
        
        sensational_words = ['shocking', 'breaking', 'exclusive', 'urgent', 'secret', 'revealed', 'exposed', 'scandal']
        emotional_words = ['amazing', 'terrible', 'incredible', 'unbelievable', 'outrageous', 'disgusting']
        found_sensational = [word for word in sensational_words if word in processed_text.lower()]
        found_emotional = [word for word in emotional_words if word in processed_text.lower()]
        
        risk_level = "HIGH" if confidence > 0.9 else "MEDIUM" if confidence > 0.7 else "LOW"
        
        return {
            "prediction": final_prediction,
            "confidence": float(confidence),
            "fake_probability": float(fake_confidence),
            "true_probability": float(true_confidence),
            "article_length": len(article_text),
            "word_count": len(processed_text.split()),
            "features_analyzed": len(feature_vector),
            "sensational_words": found_sensational,
            "emotional_words": found_emotional,
            "risk_level": risk_level
        }
    except Exception as e:
        return {"error": f"Error in prediction: {str(e)}"}

# Enhanced HTML template with modern design
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Fake News Detector</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .gradient-bg {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .glass-effect {
            background: rgba(255, 255, 255, 0.25);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }
        .animate-pulse-slow {
            animation: pulse 3s infinite;
        }
        .result-card {
            transform: translateY(20px);
            opacity: 0;
            animation: slideUp 0.6s ease-out forwards;
        }
        @keyframes slideUp {
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 24px;
            height: 24px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .progress-bar {
            transition: width 0.6s ease-in-out;
        }
        .word-count {
            transition: color 0.3s ease;
        }
    </style>
</head>
<body class="gradient-bg min-h-screen py-8 px-4">
    <div class="max-w-4xl mx-auto">
        <!-- Header -->
        <div class="text-center mb-8">
            <div class="inline-flex items-center justify-center w-20 h-20 bg-white/20 rounded-full mb-4">
                <i class="fas fa-search text-3xl text-white"></i>
            </div>
            <h1 class="text-4xl font-bold text-white mb-2">AI Fake News Detector</h1>
            <p class="text-white/80 text-lg">Advanced machine learning analysis to verify news authenticity</p>
        </div>

        <!-- Main Card -->
        <div class="glass-effect rounded-2xl p-8 shadow-2xl">
            <form id="newsForm" action="/predict/" method="post" class="space-y-6">
                <div>
                    <label for="article_text" class="block text-white font-semibold mb-3 text-lg">
                        <i class="fas fa-newspaper mr-2"></i>
                        Enter News Article Text
                    </label>
                    <div class="relative">
                        <textarea 
                            id="article_text" 
                            name="article_text" 
                            rows="8" 
                            class="w-full rounded-xl border-0 bg-white/90 backdrop-blur-sm p-4 text-gray-800 placeholder-gray-500 focus:ring-4 focus:ring-white/50 focus:outline-none transition-all duration-300 resize-none shadow-lg"
                            placeholder="Paste your news article here for analysis..."
                            oninput="updateWordCount()"
                        ></textarea>
                        <div class="absolute bottom-3 right-3 text-sm text-gray-500">
                            <span id="wordCount" class="word-count">0</span> words
                        </div>
                    </div>
                </div>
                
                <button 
                    type="submit" 
                    id="analyzeBtn"
                    class="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-bold py-4 px-8 rounded-xl transition-all duration-300 transform hover:scale-105 hover:shadow-xl flex items-center justify-center text-lg"
                >
                    <i class="fas fa-brain mr-3"></i>
                    <span id="btnText">Analyze Article</span>
                    <div id="spinner" class="spinner ml-3 hidden"></div>
                </button>
            </form>

            {% if result %}
            <div class="result-card mt-8 space-y-6">
                <!-- Main Result -->
                <div class="bg-white/90 backdrop-blur-sm rounded-xl p-6 shadow-lg">
                    <div class="flex items-center justify-between mb-4">
                        <h2 class="text-2xl font-bold text-gray-800">
                            <i class="fas fa-chart-line mr-2 text-blue-600"></i>
                            Analysis Results
                        </h2>
                        {% if result.prediction == "FAKE" %}
                        <div class="bg-red-100 text-red-800 px-4 py-2 rounded-full font-semibold flex items-center">
                            <i class="fas fa-exclamation-triangle mr-2"></i>
                            FAKE NEWS
                        </div>
                        {% else %}
                        <div class="bg-green-100 text-green-800 px-4 py-2 rounded-full font-semibold flex items-center">
                            <i class="fas fa-check-circle mr-2"></i>
                            LEGITIMATE
                        </div>
                        {% endif %}
                    </div>

                    <!-- Confidence Meter -->
                    <div class="mb-6">
                        <div class="flex justify-between items-center mb-2">
                            <span class="text-gray-700 font-medium">Confidence Level</span>
                            <span class="text-2xl font-bold text-gray-800">{{ "{:.1%}".format(result.confidence) }}</span>
                        </div>
                        <div class="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
                            {% if result.prediction == "FAKE" %}
                            <div class="progress-bar bg-gradient-to-r from-red-500 to-red-600 h-4 rounded-full shadow-inner" 
                                 style="width: {{ (result.confidence * 100)|round|int }}%"></div>
                            {% else %}
                            <div class="progress-bar bg-gradient-to-r from-green-500 to-green-600 h-4 rounded-full shadow-inner" 
                                 style="width: {{ (result.confidence * 100)|round|int }}%"></div>
                            {% endif %}
                        </div>
                    </div>

                    <!-- Risk Assessment -->
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                        <div class="bg-gradient-to-br from-red-50 to-red-100 p-4 rounded-lg border border-red-200">
                            <div class="text-red-600 font-semibold mb-1">Fake Probability</div>
                            <div class="text-2xl font-bold text-red-700">{{ "{:.1%}".format(result.fake_probability) }}</div>
                        </div>
                        <div class="bg-gradient-to-br from-green-50 to-green-100 p-4 rounded-lg border border-green-200">
                            <div class="text-green-600 font-semibold mb-1">True Probability</div>
                            <div class="text-2xl font-bold text-green-700">{{ "{:.1%}".format(result.true_probability) }}</div>
                        </div>
                        <div class="bg-gradient-to-br from-yellow-50 to-yellow-100 p-4 rounded-lg border border-yellow-200">
                            <div class="text-yellow-600 font-semibold mb-1">Risk Level</div>
                            <div class="text-2xl font-bold 
                                {% if result.risk_level == 'HIGH' %}text-red-700
                                {% elif result.risk_level == 'MEDIUM' %}text-yellow-700
                                {% else %}text-green-700{% endif %}">
                                {{ result.risk_level }}
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Detailed Analysis -->
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <!-- Article Stats -->
                    <div class="bg-white/90 backdrop-blur-sm rounded-xl p-6 shadow-lg">
                        <h3 class="text-xl font-bold text-gray-800 mb-4">
                            <i class="fas fa-file-text mr-2 text-blue-600"></i>
                            Article Statistics
                        </h3>
                        <div class="space-y-3">
                            <div class="flex justify-between items-center">
                                <span class="text-gray-600">Characters</span>
                                <span class="font-semibold text-gray-800">{{ "{:,}".format(result.article_length) }}</span>
                            </div>
                            <div class="flex justify-between items-center">
                                <span class="text-gray-600">Words</span>
                                <span class="font-semibold text-gray-800">{{ "{:,}".format(result.word_count) }}</span>
                            </div>
                            <div class="flex justify-between items-center">
                                <span class="text-gray-600">Features Analyzed</span>
                                <span class="font-semibold text-gray-800">{{ result.features_analyzed }}</span>
                            </div>
                        </div>
                    </div>

                    <!-- Warning Indicators -->
                    <div class="bg-white/90 backdrop-blur-sm rounded-xl p-6 shadow-lg">
                        <h3 class="text-xl font-bold text-gray-800 mb-4">
                            <i class="fas fa-flag mr-2 text-red-600"></i>
                            Warning Indicators
                        </h3>
                        
                        {% if result.sensational_words %}
                        <div class="mb-4">
                            <div class="text-sm font-medium text-red-600 mb-2">Sensational Words Found:</div>
                            <div class="flex flex-wrap gap-2">
                                {% for word in result.sensational_words %}
                                <span class="bg-red-100 text-red-800 px-3 py-1 rounded-full text-sm font-medium">{{ word }}</span>
                                {% endfor %}
                            </div>
                        </div>
                        {% endif %}

                        {% if result.emotional_words %}
                        <div class="mb-4">
                            <div class="text-sm font-medium text-orange-600 mb-2">Emotional Words Found:</div>
                            <div class="flex flex-wrap gap-2">
                                {% for word in result.emotional_words %}
                                <span class="bg-orange-100 text-orange-800 px-3 py-1 rounded-full text-sm font-medium">{{ word }}</span>
                                {% endfor %}
                            </div>
                        </div>
                        {% endif %}

                        {% if not result.sensational_words and not result.emotional_words %}
                        <div class="text-green-600 flex items-center">
                            <i class="fas fa-check-circle mr-2"></i>
                            No obvious warning indicators detected
                        </div>
                        {% endif %}
                    </div>
                </div>

                <!-- Disclaimer -->
                <div class="bg-blue-50 border border-blue-200 rounded-xl p-4 mt-6">
                    <div class="flex items-start">
                        <i class="fas fa-info-circle text-blue-500 mt-1 mr-3"></i>
                        <div class="text-blue-800 text-sm">
                            <strong>Disclaimer:</strong> This analysis is based on machine learning algorithms and should be used as a reference tool only. Always verify information from multiple credible sources before making conclusions about news authenticity.
                        </div>
                    </div>
                </div>
            </div>
            {% endif %}
        </div>

        <!-- Footer -->
        <div class="text-center mt-8 text-white/70">
            <p>Powered by Advanced Machine Learning • Built with FastAPI & Tailwind CSS</p>
        </div>
    </div>

    <script>
        function updateWordCount() {
            const text = document.getElementById('article_text').value;
            const wordCount = text.trim() === '' ? 0 : text.trim().split(/\s+/).length;
            const counter = document.getElementById('wordCount');
            counter.textContent = wordCount.toLocaleString();
            
            // Color coding based on word count
            if (wordCount < 50) {
                counter.className = 'word-count text-red-500';
            } else if (wordCount < 200) {
                counter.className = 'word-count text-yellow-500';
            } else {
                counter.className = 'word-count text-green-500';
            }
        }

        document.getElementById('newsForm').addEventListener('submit', function() {
            const btn = document.getElementById('analyzeBtn');
            const btnText = document.getElementById('btnText');
            const spinner = document.getElementById('spinner');
            
            btn.disabled = true;
            btn.classList.add('opacity-75');
            btnText.textContent = 'Analyzing...';
            spinner.classList.remove('hidden');
        });

        // Initialize word count on page load
        updateWordCount();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_form():
    return HTML_TEMPLATE.replace("{% if result %}", "<!--").replace("{% endif %}", "-->")

@app.post("/predict/", response_class=HTMLResponse)
async def predict(article_text: str = Form(...)):
    result = test_fake_news(article_text)
    if "error" in result:
        error_html = HTML_TEMPLATE.replace("{% if result %}", f"""
        <div class="result-card mt-8">
            <div class="bg-red-100 border border-red-400 text-red-700 px-6 py-4 rounded-xl flex items-center">
                <i class="fas fa-exclamation-triangle mr-3 text-xl"></i>
                <div>
                    <strong>Error:</strong> {result['error']}
                </div>
            </div>
        </div>
        """).replace("{% endif %}", "")
        return error_html
    
    from jinja2 import Template
    template = Template(HTML_TEMPLATE)
    return template.render(result=result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)