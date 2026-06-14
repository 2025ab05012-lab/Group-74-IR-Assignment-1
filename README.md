# 🗂️ Advanced Information Retrieval System

> **BITS Pilani – Assignment 1 Framework**  
> *Executed by Group 74*

An advanced, interactive Information Retrieval (IR) System built to process, index, and query the classic **CISI Dataset**. The application provides end-to-end processing from parsing raw corpus files to comparison of preprocessing pipelines, phrase queries, dictionary trees, and tolerant retrieval interfaces.

---

## 🚀 Key Features

* **Dataset Processing**: Automation to parse raw CISI collection files (`.ALL`, `.QRY`, `.REL`) into clean CSV structures.
* **Lexical Normalization Pipeline**: Interactive analysis of standard tokenization, Porter Stemming, and WordNet Lemmatizer morphological mappings.
* **Phrase Processing**: Evaluation of Biword Index vs. Positional Index, highlighting false-positive leakages.
* **Data Dictionary Tree structures**: Native memory-mapped implementations of Binary Search Trees (BST) and B-Trees ($t=3$) with live search latency benchmarks.
* **Tolerant Retrieval Engine**: Wildcard pattern query expansion (using Regex) and Spelling Correction (using Levenshtein Distance).
* **Inference Lab**: Quantitative metrics and analysis on IR design choices.

---

## 📁 Project Structure

```text
Assignment 1/
├── dataset/                     # Raw CISI Dataset folder
│   ├── CISI.ALL                 # Document corpus
│   ├── CISI.QRY                 # Search queries
│   └── CISI.REL                 # Relevance mappings
├── Screenshots/                 # App UI screenshots
│   ├── P1.png                   # Dataset Upload tab view
│   ├── P2.png                   # Preprocessing Comparison tab view
│   ├── P3.png                   # Phrase Processing tab view
│   ├── P4.png                   # Dictionary Trees tab view
│   ├── P5.png                   # Tolerant Retrieval tab view
│   └── P6.png                   # Inference Lab tab view
├── prepare_dataset.py           # Parser to convert raw CISI files to CSV
├── app.py                       # Main Streamlit web application
├── requirements.txt             # Python dependencies
├── GR74 Report.docx             # Detailed project report (Word)
├── GR74 Report.pdf              # Detailed project report (PDF)
└── README.md                    # Project documentation
```

---

## 🛠️ Installation & Execution

### 1. Prerequisites
Ensure you have **Python 3.8+** installed.

### 2. Install Dependencies
Install the required packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset
Parse the raw CISI dataset files into CSV tables by running:
```bash
python prepare_dataset.py
```
This generates three cleaned dataset tables:
* `cisi_docs.csv`: Document repository mappings (`doc_id`, `title`, `text`)
* `cisi_queries.csv`: System query mappings (`query_id`, `query_text`)
* `cisi_rel.csv`: Document-Query relevance mapping (`query_id`, `doc_id`)

### 4. Run the Streamlit Application
Start the interactive local dashboard:
```bash
streamlit run app.py
```

---

## 🖥️ Application Tour (Tabs & Functionality)

The application interface is divided into 6 interactive tabs:

### 📥 1. Dataset Upload
Upload the parsed `cisi_docs.csv` to populate the application's document matrix. Displays statistics such as total word population and average document depth.
* *Visual View:* `Screenshots/P1.png`

### ⚙️ 2. Preprocessing Comparison
Inspect text normalization stages (Standard, Porter Stemmer, WordNet Lemmatization) on individual document vectors. Computes collection-wide vocabulary optimization metrics.
* *Visual View:* `Screenshots/P2.png`

### 🔗 3. Phrase Processing
Compare phrase search accuracy using bigram (Biword) indexing versus positional proximity indexing. Demonstrates where the Biword index lets false positives leak.
* *Visual View:* `Screenshots/P3.png`

### 🌲 4. Dictionary Trees
Benchmark lookup speeds between a Binary Search Tree (BST) and a B-Tree ($t=3$) index.
* *Visual View:* `Screenshots/P4.png`

### 🎯 5. Tolerant Retrieval
* **Wildcard Engine**: Enter wildcard patterns (e.g., `comput*`) to query matched terms.
* **Spelling Correction**: Input out-of-vocabulary terms to find the closest matches using Levenshtein distance.
* *Visual View:* `Screenshots/P5.png`

### 📊 6. Inference Lab
Comprehensive discussion and research findings regarding the assignment questions (preprocessing selection, stemming vs. lemmatization, B-Tree advantages, and system optimizations).
* *Visual View:* `Screenshots/P6.png`

---

## 💡 System Algorithms & Core Implementations

### Levenshtein Edit Distance
Calculates the minimum single-character edits (insertions, deletions, substitutions) required to transform one word into another. Utilized in the spelling correction module:
$$\text{dist}(s_1, s_2) = \min(\text{insertions}, \text{deletions}, \text{substitutions})$$

### Binary Search Tree (BST) vs B-Tree ($t=3$)
* **BST**: A binary node tree showing $O(h)$ height searches. Can degrade to $O(n)$ if keys are added in sorted order.
* **B-Tree**: A self-balancing search tree holding multiple keys per node, keeping tree depth shallow ($O(\log_t n)$) for consistent disk/memory lookup performance.

### Positional Indexing
Indexes word terms with document lists mapping exact positions:
$$\text{Term} \rightarrow \{ \text{DocID}: [pos_1, pos_2, \dots] \}$$
Allows exact phrase verification by matching relative offsets, preventing false positives.

---

## 👥 Contributors

* **PRATA VENKATA NAGA JAYANTH RAJ** ([2025ab05012@wilp.bits-pilani.ac.in](mailto:2025ab05012@wilp.bits-pilani.ac.in))
* **ASHWIN KANTH** ([2025ab05002@wilp.bits-pilani.ac.in](mailto:2025ab05002@wilp.bits-pilani.ac.in))
* **SAROJ KUMAR MISHRA** ([2025aa05997@wilp.bits-pilani.ac.in](mailto:2025aa05997@wilp.bits-pilani.ac.in))

*Group 74 – BITS Pilani*

