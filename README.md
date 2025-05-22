# 🧠 Research Paper Title Generator using NLP

This repository contains a lightweight Natural Language Processing (NLP)-based Python tool that automatically generates meaningful and publication-ready research paper titles from scientific abstracts. The system uses syntactic analysis, TF-IDF phrase extraction, and domain-specific heuristics to extract the most informative phrases and construct high-quality titles—without the need for heavy machine learning models or training data.

---

## 🚀 Features

- 🏷️ Automatically generates academic-style titles from abstracts  
- 📚 Extracts important noun phrases and domain-relevant terms  
- 🔍 Utilizes spaCy for linguistic parsing and Scikit-learn for TF-IDF computation  
- 🧠 Filters out irrelevant or weak phrases using syntactic and semantic rules  
- 📦 Lightweight, fast, and easy to use with no deep learning dependencies  
- 🛠️ Provides fallback mechanisms for abstracts with sparse data

---

## 📌 Example

```python
abstract = """
This paper explores the use of deep learning in the automatic classification of biomedical literature.
We propose a transformer-based architecture trained on domain-specific datasets.
Experimental results demonstrate high accuracy over conventional models.
"""

title = generate_final_title(abstract)
print("Generated Title:", title)
