import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import sys
    print("SpaCy model not found. Please install it with: python -m spacy download en_core_web_sm")
    sys.exit(1)

def is_valid_phrase(phrase):
    doc = nlp(phrase)
    
    if len(doc) > 0 and doc[-1].pos_ in ['ADJ', 'ADV']:
        return False
    
    has_content = any(token.pos_ in ['NOUN', 'PROPN', 'VERB'] for token in doc)
    
    non_content_count = sum(1 for token in doc if token.is_stop or token.pos_ in ['DET', 'ADP', 'CCONJ'])
    
    return has_content and (non_content_count < len(doc) / 2)

def extract_clean_phrases(text, num_phrases=4):
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    doc = nlp(text)
    
    noun_phrases = []
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) >= 2 and len(chunk.text.split()) <= 4:
            noun_phrases.append(chunk.text.lower())
    
    words = [token.text.lower() for token in doc 
             if not token.is_stop and not token.is_punct 
             and token.pos_ in ['NOUN', 'PROPN', 'ADJ'] 
             and len(token.text) > 3]
    
    word_freq = Counter(words)
    important_words = {word for word, count in word_freq.most_common(20)}
    
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(2, 3),
        max_features=100
    )
    
    if not text or len(text.split()) < 4:
        return ["Deep Learning", "Neural Networks", "Data Analysis", "Classification"]
    
    sentences = [sent.text for sent in doc.sents]
    if not sentences:
        sentences = [text]
        
    tfidf_matrix = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names_out()
    
    tfidf_phrases = []
    for i, sent in enumerate(sentences):
        if i >= tfidf_matrix.shape[0]:
            continue
        row = tfidf_matrix[i].toarray()[0]
        top_indices = np.argsort(row)[::-1][:5]  
        for idx in top_indices:
            if row[idx] > 0:
                tfidf_phrases.append(feature_names[idx])
    
    candidate_phrases = list(set(noun_phrases + tfidf_phrases))
    
    filtered_phrases = []
    used_words = set()
    
    for phrase in candidate_phrases:
        if phrase in ['this paper', 'our approach', 'the results', 'this study']:
            continue
            
        phrase_doc = nlp(phrase)
        
        phrase_important_words = [token.text.lower() for token in phrase_doc 
                                 if token.text.lower() in important_words]
        if not phrase_important_words:
            continue
            
        clean_words = []
        for token in phrase_doc:
            if not ((token.i == 0 or token.i == len(phrase_doc)-1) and token.is_stop):
                clean_words.append(token.text)
                
        if not clean_words:
            continue
            
        clean_phrase = ' '.join(clean_words).strip()
        
        
        clean_phrase = ' '.join(word.capitalize() if word.lower() not in 
                      {'a', 'an', 'the', 'in', 'on', 'at', 'for', 'with', 'and', 'or', 'of', 'by', 'to'} 
                      else word.lower() for word in clean_phrase.split())
        
        if len(clean_phrase.split()) < 2:
            continue
            
        words = set(clean_phrase.lower().split())
        word_overlap = used_words & words
        
        if len(word_overlap) < len(words) * 0.5:
            filtered_phrases.append(clean_phrase)
            used_words.update(words)
            
        if len(filtered_phrases) >= num_phrases:
            break
    
    if len(filtered_phrases) < num_phrases:
        entities = [ent.text for ent in doc.ents if len(ent.text.split()) <= 3]
        pos_patterns = []
        for i, token in enumerate(doc):
            if token.pos_ == 'ADJ' and i+1 < len(doc) and doc[i+1].pos_ in ['NOUN', 'PROPN']:
                term = token.text + ' ' + doc[i+1].text
                pos_patterns.append(term)
        
        additional_terms = list(set(entities + pos_patterns))
        for term in additional_terms:
            if len(filtered_phrases) >= num_phrases:
                break
                
            term = ' '.join(word.capitalize() if word.lower() not in 
                  {'a', 'an', 'the', 'in', 'on', 'at', 'for', 'with', 'and', 'or', 'of', 'by', 'to'} 
                  else word.lower() for word in term.split())
                
            if term and term not in filtered_phrases:
                filtered_phrases.append(term)
    
    domain_terms = ["Deep Learning", "Neural Networks", "Machine Learning", 
                   "Data Analysis", "Classification Models"]
    
    while len(filtered_phrases) < num_phrases:
        for term in domain_terms:
            if term not in filtered_phrases:
                filtered_phrases.append(term)
                break
    
    return filtered_phrases[:num_phrases]
    
   

def generate_final_title(abstract):
    phrases = extract_clean_phrases(abstract, num_phrases=5)
    
    while len(phrases) < 5:
        phrases.append(["Deep Learning", "Machine Learning", "Data Analysis", "Experimental Results", "Methodology"][len(phrases)])
    
    doc = nlp(abstract)
    
    chunks = list(doc.noun_chunks)
    main_topics = []
    for chunk in chunks:
        if chunk.root.dep_ in ["nsubj", "dobj", "pobj"] and len(chunk.text.split()) >= 2:
            main_topics.append(chunk.text)
    
 
    verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
    common_verbs = ["propose", "present", "develop", "introduce", "implement"]
    main_verb = next((v for v in verbs if v in common_verbs), "Advancing")
    
    if main_verb.lower() == "propose":
        return f" A Novel Approach for {phrases[1]} in {phrases[2]}"
    else:
        comparison_words = ["compare", "versus", "against", "than", "performance"]
        if any(word in abstract.lower() for word in comparison_words):
            return f"{phrases[0]}: {phrases[1]} Performance in {phrases[2]}"
        else:
            return f"{phrases[0]} for {phrases[1]} in {phrases[2]}"

if __name__ == "__main__":
    abstract = """
    This paper explores the use of deep learning in the automatic classification of biomedical literature.
    We propose a transformer-based architecture trained on domain-specific datasets.
    Experimental results demonstrate high accuracy over conventional models.
    """
    
    title = generate_final_title(abstract)
    print("Generated Title:", title)
