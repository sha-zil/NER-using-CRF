import nltk
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split

# Class to parse locally available CoNLL2003 corpus
from nltk.corpus import ConllCorpusReader


# Load the dataset
def load_data():
    # Load CoNLL2003 English dataset
    data = ConllCorpusReader(r'C:\Users\shazi\OneDrive - vit.ac.in\sem7\slp\da2',['eng.train'],['words', 'pos', 'ignore', 'chunk'])
    # Retrieve sentences in IOB1 (Inside-Outside-Beginning) format
    sent_data = data.iob_sents()
    # Return tuple of word, its part-of-speech tag, and the named entity label for each word in each sentence
    return [[(word, pos, label) for word, pos, label in sent] for sent in sent_data]


# Feature extraction function
def word2features(sent, i):
    # Extract the word
    word = sent[i][0]
    # Extract the POS tag of the word
    postag = sent[i][1]
    # Extracted features:
    features = {
        'bias': 1.0,  # Constant bias added to all features
        'word.lower()': word.lower(),  # Word in lowercase
        'word[-3:]': word[-3:],  # Last 3 characters
        'word[-2:]': word[-2:],  # Last 2 characters
        'word.isupper()': word.isupper(),  # Check if uppercase
        'word.istitle()': word.istitle(),  # Check if capitalized
        'word.isdigit()': word.isdigit(),  # Check if digits
        'postag': postag,  # POS tag of word
        'postag[:2]': postag[:2],  # First two characters of POS tag
    }
    # If current word is not the first word in the sentence
    if i > 0:
        # Extract the previous word
        word1 = sent[i - 1][0]
        # Extract the POS tag of the previous word
        postag1 = sent[i - 1][1]
        # Add to features:
        features.update({
            '-1:word.lower()': word1.lower(),  # Previous word in lowercase
            '-1:word.istitle()': word1.istitle(),  # Check if previous word is capitalized
            '-1:word.isupper()': word1.isupper(),  # Check if previous word is in uppercase
            '-1:postag': postag1,  # POS tag of previous word
            '-1:postag[:2]': postag1[:2],  # First two characters of POS tag of previous word
        })
    # Else (i.e. current word is the beginning of the sentence)
    else:
        features['BOS'] = True  # Beginning of sentence
    # If current word is not the last word in the sentence
    if i < len(sent) - 1:
        # Same as before but for the next word
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    # Else (i.e. current word is the ending of the sentence)
    else:
        features['EOS'] = True  # End of sentence
    # Return dictionary of extracted features
    return features


# Function that returns extracted features from each word in a sentence
def sent2features(sent):
    # Repeatedly runs word2features() for each word in the sentence and returns a combined list
    return [word2features(sent, i) for i in range(len(sent))]


# Function that returns list of the named entity labels for each word in the sentence
def sent2labels(sent):
    # Ignoring word and pos parts of the words
    return [label for _, _, label in sent]


# Function that returns list of the tokens(word) for each word in the sentence
def sent2tokens(sent):
    return [word for word, _, _ in sent]


# Load and preprocess the data
data = load_data()
X = [sent2features(s) for s in data]
y = [sent2labels(s) for s in data]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

# Train the CRF (Conditional Random Field) model
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True,
)

# Fit the model with training data
crf.fit(X_train, y_train)

# Make predictions with testing data
y_pred = crf.predict(X_test)

# Evaluate the model
print("F1 Score:", metrics.flat_f1_score(y_test, y_pred, average='weighted'))

# Infinite loop
while True:
    # Ask for user input
    text = input("Enter a sentence: ")

    # Tokenize the sentence (split into words)
    step1 = nltk.word_tokenize(text)  # List of all words (as strings) in the sentence
    # Get POS tag for each token (word)
    step2 = nltk.pos_tag(step1)  # List of tuples each containing token and corresponding POS tag for each word
    # Format step2 for use in feature extraction function
    sent = []
    for word, pos in step2:
        sent.append((word, pos, ''))  # Finally convert each word of the sentence into (word,pos,label) format

    # Extract features from the sentence
    sentence_features = sent2features(sent)

    # Predict named entities
    predicted_labels = crf.predict_single(sentence_features)

    # Print the named entities in the sentence
    for word, label in zip(step1, predicted_labels):
        print(f"Word: {word}, Predicted Label: {label}")
