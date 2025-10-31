# 🚀 JAMBA: Hybrid Mamba-Transformer Architecture

> A cutting-edge implementation of the Jamba model, combining the efficiency of Mamba with the power of Transformer attention mechanisms.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture & Motivation](#architecture--motivation)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)
- [Resources](#resources)

---

## 🎯 Overview

JAMBA is an innovative deep learning architecture that harnesses the strengths of both **Mamba** (state-space models) and **Transformer** (attention mechanisms) to create a hybrid model. This implementation provides efficient sequence modeling with improved performance on various NLP and time-series tasks.

The project explores the synergy between:
- **Linear-time Mamba blocks** for efficient sequence processing
- **Self-attention layers** for capturing complex dependencies
- **Hybrid design patterns** for optimal performance-efficiency trade-offs

---

## 🏗️ Architecture & Motivation

### Why JAMBA?

Traditional Transformers have quadratic time complexity in sequence length, making them computationally expensive for long sequences. Mamba addresses this with linear-time complexity, but may miss some long-range dependencies that attention excels at capturing.

JAMBA combines the best of both worlds:

| Aspect | Mamba | Transformer | JAMBA |
|--------|-------|-------------|-------|
| **Time Complexity** | O(n) | O(n²) | O(n) |
| **Long-range Deps** | ✓ | ✓✓ | ✓✓ |
| **Efficiency** | ✓✓ | ✓ | ✓✓ |
| **Adaptability** | ✓ | ✓✓ | ✓✓ |

### Architecture Overview

```
Input Sequence
     ↓
  ┌──┴──┐
  │ 🔀 │ [Mamba Block]
  ├──┬──┤
  │  │  │ [Transformer Attention]
  └──┴──┘
     ↓
  MLP Layer
     ↓
Output
```

---

## ✨ Main Features

- 🧠 **Hybrid Architecture**: Seamless integration of Mamba and Transformer layers
- ⚡ **Efficient Processing**: Linear time complexity for long sequences
- 🎓 **Production-Ready**: Well-documented, tested, and optimized code
- 📊 **Flexible Configuration**: Easily customizable model dimensions and layer composition
- 🔬 **Research-Friendly**: Detailed implementation with educational value
- 💾 **Multiple Formats**: Both `.py` and Jupyter notebook implementations
- 🎯 **State-of-the-Art**: Incorporates latest research in efficient architectures

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Nakshatra1729yuvi/JAMBA.git
   cd JAMBA
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv jamba_env
   source jamba_env/bin/activate  # On Windows: jamba_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### Using JAMBA.py

```python
from JAMBA import JAMBAModel

# Initialize the model
model = JAMBAModel(
    vocab_size=50000,
    d_model=512,
    num_mamba_layers=4,
    num_transformer_layers=4,
    nhead=8,
    dim_feedforward=2048,
    max_seq_length=1024,
    dropout=0.1
)

# Forward pass
input_ids = torch.randint(0, 50000, (batch_size, seq_length))
output = model(input_ids)
print(output.shape)  # torch.Size([batch_size, seq_length, d_model])
```

### Using Jamba_Hybrid_Mamba_Transformer_Arch.ipynb

For interactive exploration and experimentation:

1. **Open the notebook**
   ```bash
   jupyter notebook Jamba_Hybrid_Mamba_Transformer_Arch.ipynb
   ```

2. **Key sections in the notebook**
   - 📚 Architecture explanation with visualizations
   - 🔧 Component implementation details
   - 🧪 Testing and validation examples
   - 📈 Performance benchmarks
   - 🎨 Visualization of attention patterns

3. **Run experiments**
   - Modify hyperparameters in the configuration cells
   - Execute cells sequentially to train and evaluate the model
   - Use provided utility functions for custom experiments

### Basic Example

```python
import torch
from JAMBA import JAMBAModel

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model
model = JAMBAModel(
    vocab_size=10000,
    d_model=256,
    num_mamba_layers=2,
    num_transformer_layers=2,
    nhead=4
).to(device)

# Create sample input
batch_size, seq_length = 8, 128
input_ids = torch.randint(0, 10000, (batch_size, seq_length)).to(device)

# Forward pass
with torch.no_grad():
    output = model(input_ids)
    print(f'Output shape: {output.shape}')
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
```

---

## 📂 Project Structure

```
JAMBA/
├── JAMBA.py                                      # Main model implementation
├── Jamba_Hybrid_Mamba_Transformer_Arch.ipynb   # Interactive notebook
├── requirements.txt                             # Python dependencies
├── README.md                                    # This file
├── LICENSE                                      # MIT License
└── docs/                                        # Additional documentation
    └── architecture.md
```

---

## 📋 Requirements

### Core Dependencies

```
torch>=2.0.0
numpy>=1.21.0
scipy>=1.7.0
tqdm>=4.62.0
```

### Optional Dependencies

```
jupyter>=1.0.0          # For notebook experiments
matplotlib>=3.5.0       # For visualizations
scikit-learn>=1.0.0     # For evaluation metrics
```

### Install with pip

```bash
pip install torch numpy scipy tqdm jupyter matplotlib scikit-learn
```

### Install from requirements file

```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

Contributions are welcome! Here's how to contribute:

1. **Fork the repository**
   ```bash
   git clone https://github.com/Nakshatra1729yuvi/JAMBA.git
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes** and commit
   ```bash
   git add .
   git commit -m "Add your descriptive commit message"
   ```

4. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request** on the main repository

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### MIT License Summary

✅ **You can:**
- Use commercially
- Modify the code
- Distribute
- Use privately

❌ **You cannot:**
- Hold the authors liable
- Use trademark

⚠️ **You must:**
- Include the license notice
- Include the copyright notice

---

## 👤 Author

**Nakshatra1729yuvi**

- 🔗 GitHub: [@Nakshatra1729yuvi](https://github.com/Nakshatra1729yuvi)
- 💼 Portfolio: [Your Portfolio/Website]
- 📧 Contact: [Your Contact Information]

---

## 📚 Resources & References

### Learn More

- 📖 [Original Mamba Paper](https://arxiv.org/abs/2312.00752)
- 📖 [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- 🔗 [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- 🔬 [Research Papers on State-Space Models](https://arxiv.org/list/cs.LG/recent)

### Related Projects

- [Mamba Implementation](https://github.com/state-spaces/mamba)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)

---

## 🔔 Updates & News

Stay updated with the latest improvements and features:

- ⭐ Star this repository if you find it useful!
- 👁️ Watch for updates
- 🔔 Get notifications for new releases

---

<div align="center">

**Made with ❤️ by Nakshatra1729yuvi**

[⬆ back to top](#-jamba-hybrid-mamba-transformer-architecture)

</div>
