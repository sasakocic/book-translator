<div align="center">
  <img src="https://github.com/user-attachments/assets/d64bd0df-b002-4365-8967-18884a5f2024" alt="logo">
  <br>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.7+-blue.svg" alt="Python 3.7+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</div>

<div align="center">
  <p>Book Translator</p>
  <p>A platform for translating books and large text documents.</p>
  <p><strong>Two-step process. Better quality.</strong></p>
</div>
  <p>The tool processes plain text files using Google Translate and Ollama LLM models. It combines primary machine translation with literary editing for better results.</p>

Support for multiple languages including English, Russian, German, French, Spanish, Italian, Chinese, and Japanese, real-time translation progress tracking for both stages, translation history and status monitoring, automatic error recovery and retry mechanisms, and real-time metrics and system monitoring.

![Book Translator Demo](https://raw.githubusercontent.com/KazKozDev/book-translator/main/demo.jpg)

## Translation Process

The application uses a sophisticated two-stage translation approach:

### Stage 1: Initial Translation
- Uses Google Translate API for fast initial translation
- Handles large volumes of text efficiently
- Provides basic translation quality
- Progress tracking for initial translation stage

### Stage 2: Literary Refinement
- Uses Ollama AI models to refine the initial translation
- Improves literary quality and natural language flow
- Maintains context and style
- Separate progress tracking for refinement stage

## Prerequisites

- Python 3.7+
- [Ollama](https://ollama.ai/) installed and running
- Node.js (for development)

## Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/kazkozdev/book-translator.git
cd book-translator
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Pull required Ollama model**
```bash
ollama pull gemma2:27b
```

4. **Start the application**
```bash
python translator.py
```

5. **Access the application**
- Open `http://localhost:5001` in your browser

## Architecture

```
book-translator/
├── translator.py        # Flask backend
├── static/             # Frontend files
├── uploads/            # Temporary uploads
├── translations/       # Completed translations
├── logs/              # Application logs
├── translations.db     # Main database
└── cache.db           # Cache database
```

## License

MIT License - see [LICENSE](LICENSE)

---
If you like this project, please give it a star ⭐

For questions, feedback, or support, reach out to:

[Artem KK](https://www.linkedin.com/in/kazkozdev/) | 
