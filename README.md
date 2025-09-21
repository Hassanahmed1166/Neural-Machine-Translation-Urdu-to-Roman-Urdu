# Neural Machine Translation: Urdu ↔ Roman Urdu

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-yellow)]()

A sequence-to-sequence neural machine translation system for translating between Urdu (Arabic script) and Roman Urdu (Latin script), built with PyTorch and LSTM networks.

## 🎯 Overview

This project addresses the challenge of translating between two different scripts of the same language - Urdu written in Arabic script (اردو) and Roman Urdu written in Latin characters. The system serves 200M+ Urdu speakers who use both scripts interchangeably in digital communication.

## ✨ Features

- **Bidirectional LSTM Encoder**: 2-layer encoder for better context understanding
- **Multi-layer LSTM Decoder**: 4-layer decoder for improved generation quality
- **Custom WordPiece Tokenization**: Handles cross-script vocabulary differences
- **Teacher Forcing**: 90% ratio during training for faster convergence
- **Multiple Evaluation Metrics**: Loss, Perplexity, and BLEU score evaluation
- **Gradient Clipping**: Prevents exploding gradients during training

## 🏗️ Architecture

```
Input (Urdu Script) → Encoder → Context → Decoder → Output (Roman Urdu)
                        ↓                    ↓
                 Bidirectional LSTM    4-layer LSTM
                   (2 layers)          (Unidirectional)
```

### Model Components

- **Encoder**: 2-layer bidirectional LSTM (256 hidden units per direction)
- **Decoder**: 4-layer unidirectional LSTM (512 hidden units)
- **Embedding**: 256-dimensional embeddings
- **Vocabulary**: 8,000 WordPiece tokens
- **Special Tokens**: `<pad>`, `<unk>`, `<sos>`, `<eos>`

## 📊 Dataset

- **Source**: Urdu-Roman Urdu parallel corpus https://github.com/amir9ume/urdu_ghazals_rekhta/tree/main
- **Format**: Excel file with paired sentences
- **Split**: 50% train, 25% validation, 25% test
- **Max Length**: 25 words (Urdu), 23 words (Roman Urdu)

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch pandas numpy scikit-learn nltk openpyxl
```

### Installation

```bash
git clone https://github.com/yourusername/urdu-roman-nmt.git
cd urdu-roman-nmt
pip install -r requirements.txt
```

### Usage

1. **Prepare Data**:
```python
python src/data_preprocessing.py --data_path data/urdu_roman.xlsx
```

2. **Train Model**:
```python
python src/train.py --config configs/default_config.json
```

3. **Translate Text**:
```python
python src/translate.py --input "یہ ایک مثال ہے" --model_path models/best_model.pth
```

## 📁 Project Structure

```
urdu-roman-nmt/
├── src/
│   ├── data_preprocessing.py    # Data loading and preprocessing
│   ├── tokenizer.py            # WordPiece tokenization
│   ├── model.py                # Seq2Seq model architecture
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation utilities
│   └── translate.py            # Translation interface
├── configs/
│   └── default_config.json     # Model hyperparameters
├── data/
│   └── urdu_roman.xlsx         # Dataset (add your own)
├── models/                     # Saved model checkpoints
├── notebooks/
│   └── nmt-v13-0.ipynb        # Original development notebook
├── requirements.txt            # Dependencies
├── README.md                   # This file
└── LICENSE                     # MIT License
```

## ⚙️ Configuration

Key hyperparameters in `configs/default_config.json`:

```json
{
  "model": {
    "vocab_size": 8000,
    "embed_dim": 256,
    "enc_hidden": 256,
    "dec_hidden": 256,
    "enc_layers": 2,
    "dec_layers": 4,
    "dropout": 0.2
  },
  "training": {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "epochs": 5,
    "teacher_forcing_ratio": 0.9,
    "grad_clip": 1.0
  }
}
```

## 📈 Current Performance

**Training Results** (5 epochs):
- **Training Loss**: Decreasing trend
- **Validation BLEU**: Showing improvement
- **Perplexity**: Reducing over epochs

*Note: This is an ongoing project. Performance metrics will be updated as improvements are made.*

## 🔄 Work in Progress

This project is actively under development. Current improvement areas:

### 🎯 Immediate Improvements
- [ ] Attention mechanism implementation
- [ ] Beam search decoding
- [ ] Advanced data augmentation
- [ ] Hyperparameter optimization

### 🚧 Future Enhancements
- [ ] Transformer architecture exploration
- [ ] Multi-head attention
- [ ] Subword regularization
- [ ] Domain adaptation techniques

## 🤝 Team

This project is being developed under the guidance of:
- **Dr. Usama** - Mentor
- **Ali Raza** - Senior Colleague  
- **Sebastian** - Collaborator
- **Hassan Ahmed** - Lead Developer

*This is my first deep dive into NLP - an incredibly exciting learning journey!*

## 📊 Evaluation Metrics

- **Perplexity**: Measures model uncertainty (lower is better)
- **BLEU Score**: N-gram overlap with reference translations
- **Loss**: Cross-entropy loss with padding ignored

## 🛠️ Technical Details

### WordPiece Tokenization
```python
def wordpiece_tokenize(word, vocab):
    chars = list(word) + ['</w>']
    tokens = []
    # Greedy longest-match-first approach
    # Falls back to <unk> for unknown subwords
```

### Training Loop
```python
for epoch in range(epochs):
    for batch in train_loader:
        outputs = model(src, trg, teacher_forcing_ratio=0.9)
        loss = criterion(outputs.view(-1, vocab_size), trg.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
```

## 🐛 Known Issues

- Long sentence translation needs improvement
- Rare words and proper nouns handling
- Context preservation in longer passages
- Script direction handling optimization

## 📚 References

- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence-to-sequence learning with neural networks
- Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📞 Contact

For questions, suggestions, or collaboration opportunities:

- **LinkedIn**: [Hassan Ahmed](https://linkedin.com/in/hassanahmed1166)

---

⭐ **Star this repository if you find it helpful!** ⭐

*Building bridges between scripts, one translation at a time.*
