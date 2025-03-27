# Book Translator

![Book Translator](https://raw.githubusercontent.com/KazKozDev/book-translator/main/banner.jpg)

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Flask](https://img.shields.io/badge/flask-%23000.svg?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![React](https://img.shields.io/badge/react-%2320232a.svg?style=flat&logo=react&logoColor=%2361DAFB)](https://reactjs.org/)
[![Tailwind CSS](https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=flat&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

Book Translator is a platform for translating books and large text documents.

Two-step process. Better quality.

The tool processes plain text files using Google Translate and Ollama LLM models. It combines primary machine translation with literary editing for better results.

## Key Features

- 🔄 Two-stage translation process:
  - Stage 1: Fast initial translation using Google Translate
  - Stage 2: Literary refinement using Ollama AI models
- 🌍 Support for multiple languages including English, Russian, German, French, Spanish, Italian, Chinese, and Japanese
- 🤖 Integration with Ollama AI models for high-quality refinements
- 🚀 Real-time translation progress tracking for both stages
- 📚 Translation history and status monitoring
- 💾 Efficient caching system for improved performance
- 🔄 Automatic error recovery and retry mechanisms
- 📊 Real-time metrics and system monitoring
- 📱 Modern, responsive UI with React and Tailwind CSS

## Demo

Experience the translation process in modern, user-friendly interface:

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

## API Reference

### Translation Endpoints
- `POST /translate` - Start translation process (both stages)
- `GET /translations` - Get history
- `GET /translations/<id>` - Get translation details including stage progress
- `GET /download/<id>` - Download refined translation
- `POST /retry-translation/<id>` - Retry failed translation

### System Endpoints
- `GET /models` - List available Ollama models
- `GET /metrics` - System metrics
- `GET /health` - Service health check
- `GET /failed-translations` - List failed translations

## Contributing

We welcome contributions! Here's how:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## License

MIT License - see [LICENSE](LICENSE) file

---

<div align="center">
  Made with ❤️
  <br>
  <a href="https://github.com/KazKozDev/book-translator/stargazers">⭐ Star us on GitHub!</a>
</div>
