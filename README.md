# Quantum-Word-Embedding
[Pre-print: TBA]
## Abstract
The accelerated progress in quantum computing has enabled a new form of machine intelligence that runs on quantum hardware, which holds great promise for more powerful computational models in various learning tasks. An emergent application of Quantum Machine Intelligence (QMI) is Quantum Natural Language Processing (QNLP). In this paper, we propose a multi-dimensional, finite automaton model for quantum word embedding (QWE) via Galois field. We demonstrated the model to three applications: (1) English vocabulary, (2) amino acid-based genetic codes, and (3) DNA-based genetic codes. The numerical results obtained from the proposed algorithm for the English vocabulary indicate that it produces more representative features for words compared to Word2Vec based on word distance metric. Second, the proposed algorithm is also utilized to model RNA-Protein interaction based on the latent distance of a given molecules, which is demonstrated on three large datasets, namely RPI369, RPI1807, and RPI2241. Finally, two embedding techniques for DNA-based genetic codes are proposed in this work, namely Two-state Lackadaisical Encoding (TCE) and Topological-Cyclic Encoding (TLE). These techniques enable relevant features to be extracted for the efficacy score of gRNAs used in the CRISPR-Cas 9 system, which is demonstrated on 15 datasets, compared to 12 mathematical features.

## Data Encoders
This is the code repository for article entitled: Quantum Word Embedding for Machine Learning.

- Encoder for English vocabulary
```
encoder_english.py
```

- Encoder for Amino-Acid-based Genetic Codes
```
encoder_aa.py
```

- Encoder for DNA-based Genetic Codes
```
encoder_dna.py
```
## Code for comparison to benchmarking Word Embedding Corpora
```
evaluate_benchmark.py
```


