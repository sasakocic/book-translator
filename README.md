![Book Translator](https://raw.githubusercontent.com/KazKozDev/book-translator/main/banner.jpg)

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Flask](https://img.shields.io/badge/flask-%23000.svg?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![React](https://img.shields.io/badge/react-%2320232a.svg?style=flat&logo=react&logoColor=%2361DAFB)](https://reactjs.org/)
[![Tailwind CSS](https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=flat&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

A web-based application for translating books and large text documents between multiple languages using Ollama AI models. Built with Python (Flask) backend and React frontend.

## Features

### Core Features
- Translate text documents between multiple languages:
  - English
  - Russian
  - German
  - French
  - Spanish
  - Italian
  - Chinese
  - Japanese
- Automatic language detection
- Support for multiple Ollama AI models
- Real-time translation progress tracking
- Translation history with status tracking
- Resume interrupted translations
- Download translated files
- Modern, responsive UI built with React and Tailwind CSS

### Advanced Features
- Translation caching system
- Automatic error recovery
- Rate limiting and retry logic
- Chunked translation processing
- Real-time metrics monitoring
- Health check system
- Comprehensive logging system
- Automatic cleanup of old translations

## Prerequisites
- Python 3.7+
- [Ollama](https://ollama.ai/) installed and running
- Node.js (optional, for development)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kazkozdev/book-translator.git
cd book-translator
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure Ollama is running and you have at least one model pulled:
```bash
ollama pull aya-expanse:32b
```

## Running the Application

1. Start the Flask backend:
```bash
python translator.py
```

2. Access the application:
Open `http://localhost:5001` in your web browser

## Project Structure
```
book-translator/
├── translator.py        # Flask backend
├── static/             # Static files
│   └── index.html      # React frontend
├── uploads/            # Temporary upload directory
├── translations/       # Completed translations directory
├── logs/              # Application logs directory
│   ├── app.log        # Main application logs
│   ├── translations.log # Translation-specific logs
│   └── api.log        # API request logs
├── translations.db     # Main SQLite database
└── cache.db           # Translation cache database
```

## API Endpoints

### Translation Operations
- `POST /translate` - Start new translation
- `GET /translations` - Get translation history
- `GET /translations/<id>` - Get specific translation details
- `GET /download/<id>` - Download completed translation
- `POST /retry-translation/<id>` - Retry failed translation

### System Operations
- `GET /models` - Get available Ollama models
- `GET /metrics` - Get system metrics and statistics
- `GET /health` - Check service health
- `GET /failed-translations` - Get list of failed translations

## Monitoring and Metrics

The application provides real-time monitoring of:
- Translation success rate
- Average translation time
- CPU usage
- Memory usage
- Disk usage
- System uptime
- Translation queue status

## Error Handling and Recovery

- Automatic retry system for failed translations
- Configurable retry attempts and backoff
- Detailed error logging and tracking
- Translation state preservation
- Automatic cleanup of failed translations after 7 days

## Caching System

- Translation results are cached for improved performance
- Configurable cache retention period (default 30 days)
- Automatic cache cleanup
- Cache hit/miss tracking

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

![Book Translator](https://raw.githubusercontent.com/KazKozDev/book-translator/main/demo.jpg)
