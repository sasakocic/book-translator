import json
import requests
import time
from typing import List, Dict, Optional, Tuple, Callable
from tqdm import tqdm
import os
import sqlite3
from datetime import datetime, timedelta
import logging
from logging.handlers import RotatingFileHandler
import hashlib
import traceback
import psutil
import threading
from collections import deque
from dataclasses import dataclass, field
from functools import wraps
from flask import Flask, request, jsonify, Response, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from deep_translator import GoogleTranslator
import unicodedata

app = Flask(__name__)
CORS(app)

# Folders setup
UPLOAD_FOLDER = 'uploads'
TRANSLATIONS_FOLDER = 'translations'
STATIC_FOLDER = 'static'
LOG_FOLDER = 'logs'
DB_PATH = 'translations.db'
CACHE_DB_PATH = 'cache.db'

# Provider URLs
OLLAMA_URL = "http://localhost:11434"
LMSTUDIO_URL = "http://localhost:1234"

# Create necessary directories
for folder in [UPLOAD_FOLDER, TRANSLATIONS_FOLDER, STATIC_FOLDER, LOG_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Logger setup
class AppLogger:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.app_logger = self._setup_logger(
            'app_logger',
            os.path.join(log_dir, 'app.log')
        )
        
        self.translation_logger = self._setup_logger(
            'translation_logger',
            os.path.join(log_dir, 'translations.log')
        )
        
        self.api_logger = self._setup_logger(
            'api_logger',
            os.path.join(log_dir, 'api.log')
        )

    def _setup_logger(self, name, log_file):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        handler = RotatingFileHandler(
            log_file,
            maxBytes=50*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        return logger

# Initialize logger
logger = AppLogger()

# Monitoring setup
@dataclass
class TranslationMetrics:
    total_requests: int = 0
    successful_translations: int = 0
    failed_translations: int = 0
    average_translation_time: float = 0
    translation_times: deque = field(default_factory=lambda: deque(maxlen=100))

class AppMonitor:
    def __init__(self):
        self.metrics = TranslationMetrics()
        self._lock = threading.Lock()
        self.start_time = time.time()
        
    def record_translation_attempt(self, success: bool, translation_time: float):
        with self._lock:
            self.metrics.total_requests += 1
            if success:
                self.metrics.successful_translations += 1
                self.metrics.translation_times.append(translation_time)
                self.metrics.average_translation_time = (
                    sum(self.metrics.translation_times) / len(self.metrics.translation_times)
                )
            else:
                self.metrics.failed_translations += 1
    
    def get_system_metrics(self) -> Dict:
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'uptime': time.time() - self.start_time
        }
    
    def get_metrics(self) -> Dict:
        with self._lock:
            metrics_data = {
                'translation_metrics': {
                    'total_requests': self.metrics.total_requests,
                    'successful_translations': self.metrics.successful_translations,
                    'failed_translations': self.metrics.failed_translations,
                    'average_translation_time': self.metrics.average_translation_time
                },
                'system_metrics': self.get_system_metrics()
            }
            
            # Calculate success rate only if there are requests
            if self.metrics.total_requests > 0:
                metrics_data['translation_metrics']['success_rate'] = (
                    self.metrics.successful_translations / self.metrics.total_requests * 100
                )
            else:
                metrics_data['translation_metrics']['success_rate'] = 0
                
            return metrics_data

# Initialize monitor
monitor = AppMonitor()

# Translation cache setup
class TranslationCache:
    def __init__(self, db_path: str = CACHE_DB_PATH):
        self.db_path = db_path
        self._init_cache_db()
    
    def _init_cache_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS translation_cache (
                    hash_key TEXT PRIMARY KEY,
                    source_lang TEXT,
                    target_lang TEXT,
                    original_text TEXT,
                    translated_text TEXT,
                    machine_translation TEXT,
                    created_at TIMESTAMP,
                    last_used TIMESTAMP
                )
            ''')

    def _generate_hash(self, text: str, source_lang: str, target_lang: str) -> str:
        key = f"{text}:{source_lang}:{target_lang}".encode('utf-8')
        return hashlib.sha256(key).hexdigest()
    
    def get_cached_translation(self, text: str, source_lang: str, target_lang: str) -> Optional[Dict[str, str]]:
        hash_key = self._generate_hash(text, source_lang, target_lang)
        
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute('''
                SELECT translated_text, machine_translation
                FROM translation_cache
                WHERE hash_key = ?
            ''', (hash_key,))
            
            result = cur.fetchone()
            if result:
                conn.execute('''
                    UPDATE translation_cache
                    SET last_used = CURRENT_TIMESTAMP
                    WHERE hash_key = ?
                ''', (hash_key,))
                return {
                    'translated_text': result[0],
                    'machine_translation': result[1]
                }
        
        return None
    
    def cache_translation(self, text: str, translated_text: str, machine_translation: str, 
                         source_lang: str, target_lang: str):
        hash_key = self._generate_hash(text, source_lang, target_lang)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO translation_cache
                (hash_key, source_lang, target_lang, original_text, translated_text, 
                 machine_translation, created_at, last_used)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ''', (hash_key, source_lang, target_lang, text, translated_text, machine_translation))
    
    def cleanup_old_entries(self, days: int = 30):
        with sqlite3.connect(self.db_path) as conn:
            # Use direct string formatting for date arithmetic since SQLite's 
            # datetime() function doesn't accept parameters for interval
            conn.execute(
                f"DELETE FROM translation_cache WHERE last_used < datetime('now', '-{days} days')"
            )

# Initialize cache
cache = TranslationCache()

# Error handling setup
class TranslationError(Exception):
    pass

def with_error_handling(f: Callable):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except requests.Timeout as e:
            logger.app_logger.error(f"Timeout error: {str(e)}")
            raise TranslationError("Translation service timeout")
        except requests.RequestException as e:
            logger.app_logger.error(f"Request error: {str(e)}")
            raise TranslationError("Translation service unavailable")
        except sqlite3.Error as e:
            logger.app_logger.error(f"Database error: {str(e)}")
            raise TranslationError("Database error occurred")
        except Exception as e:
            logger.app_logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
            raise TranslationError("An unexpected error occurred")
    return wrapper

# Initialize database
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        # Drop existing tables if they exist and recreate them
        conn.executescript('''
            DROP TABLE IF EXISTS chunks;
            DROP TABLE IF EXISTS translations;
            
            CREATE TABLE translations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                source_lang TEXT NOT NULL,
                target_lang TEXT NOT NULL,
                model TEXT NOT NULL,
                status TEXT NOT NULL,
                progress REAL DEFAULT 0,
                current_chunk INTEGER DEFAULT 0,
                total_chunks INTEGER DEFAULT 0,
                original_text TEXT,
                machine_translation TEXT,
                translated_text TEXT,
                detected_language TEXT,
                genre TEXT DEFAULT 'unknown',  -- Added genre with default value
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                error_message TEXT
            );

            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                translation_id INTEGER,
                chunk_number INTEGER,
                original_text TEXT,
                machine_translation TEXT,
                translated_text TEXT,
                status TEXT,
                error_message TEXT,
                attempts INTEGER DEFAULT 0,
                FOREIGN KEY (translation_id) REFERENCES translations (id)
            );
        ''')

init_db()

class BookTranslator:
    def __init__(self, model_name: str = "aya-expanse:32b", provider: str = "ollama", chunk_size: int = 1000):
        self.model_name = model_name
        self.provider = provider
        self.chunk_size = chunk_size
        self.total_token_usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
        
        # Set API URL based on provider
        if provider == "lmstudio":
            self.api_url = f"{LMSTUDIO_URL}/v1/chat/completions"
            self.models_url = f"{LMSTUDIO_URL}/v1/models"
        else:  # default to ollama
            self.api_url = f"{OLLAMA_URL}/api/generate"
            self.models_url = f"{OLLAMA_URL}/api/tags"
        
        # Serbian Cyrillic to Latin transliteration map
        self.cyrillic_to_latin = {
            'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Ђ': 'Đ', 'Е': 'E', 'Ж': 'Ž',
            'З': 'Z', 'И': 'I', 'Ј': 'J', 'К': 'K', 'Л': 'L', 'Љ': 'Lj', 'М': 'M', 'Н': 'N',
            'Њ': 'Nj', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'Ћ': 'Ć', 'У': 'U',
            'Ф': 'F', 'Х': 'H', 'Ц': 'C', 'Ч': 'Č', 'Џ': 'Dž', 'Ш': 'Š',
            'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'ђ': 'đ', 'е': 'e', 'ж': 'ž',
            'з': 'z', 'и': 'i', 'ј': 'j', 'к': 'k', 'л': 'l', 'љ': 'lj', 'м': 'm', 'н': 'n',
            'њ': 'nj', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'ћ': 'ć', 'у': 'u',
            'ф': 'f', 'х': 'h', 'ц': 'c', 'ч': 'č', 'џ': 'dž', 'ш': 'š'
        }
        self.session = requests.Session()
        self.session.mount('http://', requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=10,
            pool_maxsize=10
        ))

    def split_into_chunks(self, text: str, for_llm: bool = False) -> list:
        """Split text into chunks for translation.
        
        Args:
            text: Text to split
            for_llm: If True, use larger chunks optimized for 128K token LLMs
        """
        if for_llm:
            # Optimize for 128K token models (~512K characters)
            # Use ~100K characters per chunk to stay well within limits
            MAX_LENGTH = 100000
        else:
            MAX_LENGTH = 4500  # Google Translate limit
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            if len(paragraph) + current_length > MAX_LENGTH:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split long paragraphs if needed
                if len(paragraph) > MAX_LENGTH:
                    sentences = paragraph.split('. ')
                    temp_chunk = []
                    temp_length = 0
                    
                    for sentence in sentences:
                        if temp_length + len(sentence) > MAX_LENGTH:
                            if temp_chunk:
                                chunks.append('. '.join(temp_chunk) + '.')
                                temp_chunk = []
                                temp_length = 0
                        temp_chunk.append(sentence)
                        temp_length += len(sentence) + 2  # +2 for '. '
                        
                    if temp_chunk:
                        chunks.append('. '.join(temp_chunk) + '.')
                else:
                    current_chunk.append(paragraph)
                    current_length = len(paragraph)
            else:
                current_chunk.append(paragraph)
                current_length += len(paragraph) + 2  # +2 for '\n\n'
                
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            
        return chunks

    def transliterate_serbian(self, text: str) -> str:
        """Convert Serbian Cyrillic to Latin script"""
        result = ""
        i = 0
        while i < len(text):
            # Handle multi-character mappings first
            if i < len(text) - 1:
                two_char = text[i:i+2]
                if two_char in self.cyrillic_to_latin:
                    result += self.cyrillic_to_latin[two_char]
                    i += 2
                    continue
            
            # Handle single characters
            char = text[i]
            if char in self.cyrillic_to_latin:
                result += self.cyrillic_to_latin[char]
            else:
                result += char
            i += 1
        
        return result

    def translate_text(self, text: str, source_lang: str, target_lang: str, translation_id: int, skip_llm_refinement: bool = False, use_cache: bool = True, skip_google_translate: bool = False):
        start_time = time.time()
        success = False
        
        try:
            if skip_google_translate:
                # LLM-only mode: Use larger chunks optimized for LLM
                chunks = self.split_into_chunks(text, for_llm=True)
                total_chunks = len(chunks)
                translated_chunks = []
                machine_translations = []  # Empty for LLM-only mode
                
                logger.translation_logger.info(f"Starting LLM-only translation {translation_id} with {total_chunks} chunks")
            else:
                # Use small chunks for Google Translate (API limit)
                chunks = self.split_into_chunks(text, for_llm=False)
                total_chunks = len(chunks)
                translated_chunks = []
                machine_translations = []
                
                logger.translation_logger.info(f"Starting translation {translation_id} with {total_chunks} chunks")
                
                # Initialize Google translator
                translator = GoogleTranslator(
                    source=source_lang if source_lang != 'auto' else 'auto',
                    target=target_lang
                )
            
            # Update database with total chunks (same for both modes now)
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute('''
                    UPDATE translations 
                    SET total_chunks = ?, status = 'in_progress'
                    WHERE id = ?
                ''', (total_chunks, translation_id))
            
            for i, chunk in enumerate(chunks, 1):
                try:
                    # Check cache first (if caching is enabled)
                    cached_result = None
                    if use_cache:
                        cached_result = cache.get_cached_translation(chunk, source_lang, target_lang)
                    
                    if cached_result:
                        if not skip_google_translate:
                            machine_translations.append(cached_result['machine_translation'])
                        translated_chunks.append(cached_result['translated_text'])
                        logger.translation_logger.info(f"Cache hit for chunk {i}")
                    elif skip_google_translate:
                        # LLM-only mode: Direct LLM translation
                        logger.translation_logger.info(f"LLM-only translating chunk {i}/{total_chunks}")
                        llm_translation = self.direct_llm_translate(chunk, source_lang, target_lang)
                        
                        # Convert Serbian Cyrillic to Latin if needed
                        if target_lang.lower() == 'sr':
                            llm_translation = self.transliterate_serbian(llm_translation)
                            logger.translation_logger.info(f"Transliterated LLM output to Latin: {llm_translation[:50]}...")
                        
                        translated_chunks.append(llm_translation)
                        
                        # Cache the results (no machine translation for LLM-only)
                        if use_cache:
                            cache.cache_translation(
                                chunk, llm_translation, llm_translation,
                                source_lang, target_lang
                            )
                    else:
                        # Stage 1: Google Translate
                        logger.translation_logger.info(f"Translating chunk {i}/{total_chunks}")
                        google_translation = translator.translate(chunk)
                        
                        # Convert Serbian Cyrillic to Latin if needed
                        if target_lang.lower() == 'sr':
                            google_translation = self.transliterate_serbian(google_translation)
                            logger.translation_logger.info(f"Transliterated Serbian to Latin: {google_translation[:50]}...")
                        
                        machine_translations.append(google_translation)
                        
                        # If skipping LLM refinement, use Google translation as final result
                        if skip_llm_refinement:
                            translated_chunks.append(google_translation)
                            final_progress = (i / total_chunks) * 100
                            
                            # Cache the results (same for both machine and translated)
                            if use_cache:
                                cache.cache_translation(
                                    chunk, google_translation, google_translation,
                                    source_lang, target_lang
                                )
                            
                            progress_update = {
                                'progress': final_progress,
                                'stage': 'google_translate_only',
                                'machine_translation': '\n\n'.join(machine_translations),
                                'translated_text': '\n\n'.join(translated_chunks),
                                'current_chunk': i,
                                'total_chunks': total_chunks,
                                'token_usage': self.total_token_usage.copy()
                            }
                            
                            logger.translation_logger.info(f"Yielding Google-only progress: {final_progress:.1f}% - Chunk {i}/{total_chunks}")
                            yield progress_update
                            
                            # Small delay to ensure frontend can process the update
                            time.sleep(0.1)
                        else:
                            # Stage 2: Literary refinement
                            logger.translation_logger.info(f"Refining chunk {i}/{total_chunks}")
                            refined_translation = self.refine_translation(google_translation, target_lang)
                            
                            # Convert Serbian Cyrillic to Latin if needed (for LLM output)
                            if target_lang.lower() == 'sr':
                                refined_translation = self.transliterate_serbian(refined_translation)
                                logger.translation_logger.info(f"Transliterated LLM output to Latin: {refined_translation[:50]}...")
                            
                            translated_chunks.append(refined_translation)
                            
                            # Cache the results
                            if use_cache:
                                cache.cache_translation(
                                    chunk, refined_translation, google_translation,
                                    source_lang, target_lang
                                )
                    
                    # Update progress and database (single update per chunk)
                    if skip_google_translate:
                        # For LLM-only mode
                        progress = (i / total_chunks) * 100
                        current_chunk = i
                        total_chunks_display = total_chunks
                        stage = 'llm_only'
                    elif not skip_llm_refinement:
                        # For two-stage: each completed chunk represents 1/total_chunks of 100%
                        progress = (i / total_chunks) * 100
                        current_chunk = i
                        total_chunks_display = total_chunks
                        stage = 'literary_refinement'
                    else:
                        progress = (i / total_chunks) * 100
                        current_chunk = i
                        total_chunks_display = total_chunks
                        stage = 'google_translate_only'
                    
                    with sqlite3.connect(DB_PATH) as conn:
                        conn.execute('''
                            UPDATE translations 
                            SET progress = ?,
                                translated_text = ?,
                                machine_translation = ?,
                                current_chunk = ?,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = ?
                        ''', (
                            progress,
                            '\n\n'.join(translated_chunks),
                            '\n\n'.join(machine_translations),
                            current_chunk,
                            translation_id
                        ))
                    
                    # Send progress update (single update per chunk)
                    progress_update = {
                        'progress': progress,
                        'stage': stage,
                        'machine_translation': '\n\n'.join(machine_translations),
                        'translated_text': '\n\n'.join(translated_chunks),
                        'current_chunk': current_chunk,
                        'total_chunks': total_chunks_display,
                        'token_usage': self.total_token_usage.copy()
                    }
                    
                    logger.translation_logger.info(f"Yielding progress update: {progress:.1f}% - Chunk {current_chunk}/{total_chunks_display}")
                    yield progress_update
                    
                    # Small delay to ensure frontend can process the update
                    time.sleep(0.1)
                    
                except Exception as e:
                    error_msg = f"Error processing chunk {i}: {str(e)}"
                    logger.translation_logger.error(error_msg)
                    logger.translation_logger.error(traceback.format_exc())
                    raise Exception(error_msg)
                    
                time.sleep(1)  # Rate limiting
                
            # Optimize LLM refinement for 128K context if not skipping LLM
            if not skip_llm_refinement and len(machine_translations) > 1:
                logger.translation_logger.info("Optimizing translation with 128K context window...")
                
                # Create larger chunks for LLM processing (up to ~100K characters)
                llm_chunks = self.split_into_chunks('\n\n'.join(machine_translations), for_llm=True)
                optimized_translations = []
                
                for i, llm_chunk in enumerate(llm_chunks):
                    logger.translation_logger.info(f"LLM optimization chunk {i+1}/{len(llm_chunks)}")
                    optimized_chunk = self.refine_translation(llm_chunk, target_lang)
                    
                    # Convert Serbian Cyrillic to Latin if needed (for optimized LLM output)
                    if target_lang.lower() == 'sr':
                        optimized_chunk = self.transliterate_serbian(optimized_chunk)
                        logger.translation_logger.info(f"Transliterated optimized chunk to Latin: {optimized_chunk[:50]}...")
                    
                    optimized_translations.append(optimized_chunk)
                
                # Replace the translated_chunks with optimized version
                final_translation = '\n\n'.join(optimized_translations)
                # Split back to match original chunk structure for consistency
                translated_chunks = final_translation.split('\n\n')
                
                # Update database with optimized translation
                with sqlite3.connect(DB_PATH) as conn:
                    conn.execute('''
                        UPDATE translations 
                        SET translated_text = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    ''', ('\n\n'.join(translated_chunks), translation_id))
                
                logger.translation_logger.info("128K context optimization completed")

            # Mark translation as completed
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute('''
                    UPDATE translations 
                    SET status = 'completed',
                        progress = 100,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (translation_id,))
                
            success = True
            yield {
                'progress': 100,
                'machine_translation': '\n\n'.join(machine_translations),
                'translated_text': '\n\n'.join(translated_chunks),
                'status': 'completed',
                'token_usage': self.total_token_usage.copy()
            }
            
        except Exception as e:
            error_msg = f"Translation failed: {str(e)}"
            logger.translation_logger.error(error_msg)
            logger.translation_logger.error(traceback.format_exc())
            
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute('''
                    UPDATE translations 
                    SET status = 'error',
                        error_message = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (str(e), translation_id))
            raise
        finally:
            translation_time = time.time() - start_time
            monitor.record_translation_attempt(success, translation_time)
    
    def refine_translation(self, text: str, target_lang: str) -> str:
        """
        Refine the machine translation strictly in the target language.
        
        Args:
            text (str): The machine-translated text to refine
            target_lang (str): The target language code (e.g., 'en', 'es', 'fr')
        
        Returns:
            str: The refined translation
        """
        # Промпты для каждого языка
        prompts = {
            'en': 'Improve this text to sound more natural in English. Return only the improved text:',
            'es': 'Mejora este texto para que suene más natural en español. Devuelve solo el texto mejorado:',
            'fr': 'Améliorez ce texte pour qu\'il sonne plus naturel en français. Retournez uniquement le texte amélioré :',
            'de': 'Verbessern Sie diesen Text, damit er auf Deutsch natürlicher klingt. Geben Sie nur den verbesserten Text zurück:',
            'it': 'Migliora questo testo per renderlo più naturale in italiano. Restituisci solo il testo migliorato:',
            'pt': 'Melhore este texto para soar mais natural em português. Retorne apenas o texto melhorado:',
            'ru': 'Улучшите этот текст, чтобы он звучал более естественно на русском языке. Верните только улучшенный текст:',
            'zh': '改善这段文字，使其在中文中更加自然。仅返回改善后的文字：',
            'ja': 'この文章を日本語としてより自然に聞こえるように改善してください。改善されたテキストのみを返してください：',
            'ko': '이 텍스트를 한국어로 더 자연스럽게 들리도록 개선하십시오. 개선된 텍스트만 반환하십시오:',
            'sr': 'Poboljšajte ovaj tekst da zvuči prirodnije na srpskom jeziku. VAŽNO: Koristite SAMO latinicu (ne ćirilicu). Vratite samo poboljšani tekst:',
            'hr': 'Poboljšajte ovaj tekst da zvuči prirodnije na hrvatskom jeziku. Vratite samo poboljšani tekst:'
        }
        
        # Получаем промпт для выбранного языка или используем английский как запасной вариант
        prompt_text = prompts.get(target_lang.lower(), prompts['en'])
        
        # Debug logging for language detection
        logger.translation_logger.info(f"Refinement for language '{target_lang}' (type: {type(target_lang)}) -> Available keys: {list(prompts.keys())}")
        logger.translation_logger.info(f"Looking for key: '{target_lang.lower()}' -> Found: {'sr' in prompts}")
        logger.translation_logger.info(f"Using prompt: {prompt_text[:50]}...")
        
        if self.provider == "lmstudio":
            # LMStudio uses OpenAI-compatible API
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": f"{prompt_text}\n\n{text}"}
                ],
                "stream": False,
                "temperature": 0.7
            }
            
            response = self.session.post(
                self.api_url,
                json=payload,
                timeout=(300, 300)
            )
            response.raise_for_status()
            result = response.json()
            
            # Track token usage for LMStudio
            if 'usage' in result:
                usage = result['usage']
                self.total_token_usage['prompt_tokens'] += usage.get('prompt_tokens', 0)
                self.total_token_usage['completion_tokens'] += usage.get('completion_tokens', 0)
                self.total_token_usage['total_tokens'] += usage.get('total_tokens', 0)
                logger.translation_logger.info(f"LMStudio token usage: {usage}")
            
            return result['choices'][0]['message']['content'].strip()
        else:
            # Ollama API
            prompt = f"""{prompt_text}
    
    {text}"""
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            
            response = self.session.post(
                self.api_url,
                json=payload,
                timeout=(300, 300)
            )
            response.raise_for_status()
            result = response.json()
            
            # Track token usage for Ollama (if available)
            if 'prompt_eval_count' in result and 'eval_count' in result:
                prompt_tokens = result.get('prompt_eval_count', 0)
                completion_tokens = result.get('eval_count', 0)
                total_tokens = prompt_tokens + completion_tokens
                
                self.total_token_usage['prompt_tokens'] += prompt_tokens
                self.total_token_usage['completion_tokens'] += completion_tokens
                self.total_token_usage['total_tokens'] += total_tokens
                
                logger.translation_logger.info(f"Ollama token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
            
            return result['response'].strip()
    
    def direct_llm_translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Direct LLM translation without Google Translate preprocessing.
        
        Args:
            text (str): The original text to translate
            source_lang (str): The source language code
            target_lang (str): The target language code
        
        Returns:
            str: The translated text
        """
        # Simplified, clear prompts for direct translation
        prompts = {
            'en': f'Translate from {source_lang} to English. Keep the same formatting and structure. Only return the translation:\n\n',
            'es': f'Traduce de {source_lang} al español. Mantén el mismo formato y estructura. Solo devuelve la traducción:\n\n',
            'fr': f'Traduisez de {source_lang} vers le français. Gardez le même format et structure. Retournez seulement la traduction:\n\n',
            'de': f'Übersetze von {source_lang} ins Deutsche. Behalte das gleiche Format und die Struktur bei. Gib nur die Übersetzung zurück:\n\n',
            'it': f'Traduci da {source_lang} all\'italiano. Mantieni lo stesso formato e struttura. Restituisci solo la traduzione:\n\n',
            'pt': f'Traduza de {source_lang} para português. Mantenha o mesmo formato e estrutura. Retorne apenas a tradução:\n\n',
            'ru': f'Переведи с {source_lang} на русский. Сохрани тот же формат и структуру. Верни только перевод:\n\n',
            'zh': f'从{source_lang}翻译成中文。保持相同的格式和结构。只返回翻译：\n\n',
            'ja': f'{source_lang}から日本語に翻訳してください。同じ形式と構造を保ってください。翻訳のみを返してください：\n\n',
            'ko': f'{source_lang}에서 한국어로 번역하세요. 같은 형식과 구조를 유지하세요. 번역만 반환하세요:\n\n',
            'sr': f'Prevedi sa {source_lang} na srpski (latinica). Zadrži isti format i strukturu. Vrati samo prevod:\n\n',
            'hr': f'Prevedi s {source_lang} na hrvatski. Zadrži isti format i strukturu. Vrati samo prijevod:\n\n'
        }
        
        # Get prompt for target language or use English as fallback
        prompt_text = prompts.get(target_lang.lower(), f'Translate from {source_lang} to {target_lang}. Keep the same formatting. Only return the translation:\n\n')
        
        logger.translation_logger.info(f"Direct LLM translation from '{source_lang}' to '{target_lang}'")
        logger.translation_logger.info(f"Using prompt: {prompt_text[:50]}...")
        
        if self.provider == "lmstudio":
            # LMStudio uses OpenAI-compatible API
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": f"{prompt_text}\n\n{text}"}
                ],
                "stream": False,
                "temperature": 0.7
            }
            
            response = self.session.post(
                self.api_url,
                json=payload,
                timeout=(300, 300)
            )
            response.raise_for_status()
            result = response.json()
            
            # Track token usage for LMStudio
            if 'usage' in result:
                usage = result['usage']
                self.total_token_usage['prompt_tokens'] += usage.get('prompt_tokens', 0)
                self.total_token_usage['completion_tokens'] += usage.get('completion_tokens', 0)
                self.total_token_usage['total_tokens'] += usage.get('total_tokens', 0)
                logger.translation_logger.info(f"LMStudio token usage: {usage}")
            
            return result['choices'][0]['message']['content'].strip()
        else:
            # Ollama API
            prompt = f"{prompt_text}{text}"
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            
            response = self.session.post(
                self.api_url,
                json=payload,
                timeout=(300, 300)
            )
            response.raise_for_status()
            result = response.json()
            
            # Track token usage for Ollama (if available)
            if 'prompt_eval_count' in result and 'eval_count' in result:
                prompt_tokens = result.get('prompt_eval_count', 0)
                completion_tokens = result.get('eval_count', 0)
                total_tokens = prompt_tokens + completion_tokens
                
                self.total_token_usage['prompt_tokens'] += prompt_tokens
                self.total_token_usage['completion_tokens'] += completion_tokens
                self.total_token_usage['total_tokens'] += total_tokens
                
                logger.translation_logger.info(f"Ollama token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
            
            return result['response'].strip()
    
    def get_available_models(self) -> List[str]:
        response = self.session.get(
            self.models_url,
            timeout=(5, 5)
        )
        response.raise_for_status()
        models = response.json()
        
        if self.provider == "lmstudio":
            # LMStudio returns OpenAI-compatible format
            return [model['id'] for model in models['data']]
        else:
            # Ollama format
            return [model['name'] for model in models['models']]

# Translation Recovery
class TranslationRecovery:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        
    def get_failed_translations(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute('''
                SELECT * FROM translations 
                WHERE status = 'error'
                ORDER BY created_at DESC
            ''')
            return [dict(row) for row in cur.fetchall()]
        
    def retry_translation(self, translation_id: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE translations
                SET status = 'pending', progress = 0, error_message = NULL,
                    current_chunk = 0, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (translation_id,))
            
            conn.execute('''
                UPDATE chunks
                SET status = 'pending', error_message = NULL
                WHERE translation_id = ? AND status = 'error'
            ''', (translation_id,))
            
    def cleanup_failed_translations(self, days: int = 7):
        with sqlite3.connect(self.db_path) as conn:
            # Use direct string formatting for date arithmetic since SQLite's
            # datetime() function doesn't accept parameters for interval
            conn.execute(
                f"DELETE FROM translations WHERE status = 'error' AND created_at < datetime('now', '-{days} days')"
            )

recovery = TranslationRecovery()

# Health checking middleware
@app.before_request
def check_services():
    # Skip service checks for static files and basic endpoints
    if request.endpoint in ['health_check', 'get_providers', 'serve_frontend', 'serve_static']:
        return
    
    # For translate endpoint, only check the specific provider if a model is specified
    if request.endpoint == 'translate' and request.method == 'POST':
        model_name = request.form.get('model')
        provider = request.form.get('provider', 'ollama')
        
        # If no model specified, only Google Translate will be used
        if not model_name or not model_name.strip():
            return
            
        # Check the specific provider
        try:
            if provider == 'lmstudio':
                response = requests.get(f"{LMSTUDIO_URL}/v1/models", timeout=3)
            else:  # ollama
                response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
            
            if response.status_code != 200:
                raise requests.exceptions.RequestException(f"Service returned status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.app_logger.error(f"{provider.title()} health check failed: {str(e)}")
            return jsonify({
                'error': f'{provider.title()} service is not available. You can still use Google Translate only by leaving the model field empty.'
            }), 503

# Flask routes
@app.route('/')
def serve_frontend():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/providers', methods=['GET'])
@with_error_handling
def get_providers():
    providers = []
    
    # Check Ollama
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        if response.status_code == 200:
            providers.append({'id': 'ollama', 'name': 'Ollama', 'url': OLLAMA_URL, 'status': 'available'})
    except:
        pass  # Ollama not available
    
    # Check LMStudio
    try:
        response = requests.get(f"{LMSTUDIO_URL}/v1/models", timeout=3)
        if response.status_code == 200:
            providers.append({'id': 'lmstudio', 'name': 'LM Studio', 'url': LMSTUDIO_URL, 'status': 'available'})
    except:
        pass  # LMStudio not available
    
    return jsonify({'providers': providers})

@app.route('/models', methods=['GET'])
@with_error_handling
def get_models():
    provider = request.args.get('provider', 'ollama')
    translator = BookTranslator(provider=provider)
    try:
        available_models = translator.get_available_models()
        models = []
        for model_name in available_models:
            models.append({
                'name': model_name,
                'size': 'Unknown',
                'modified': 'Unknown'
            })
        return jsonify({'models': models})
    except Exception as e:
        logger.app_logger.error(f"Error fetching models for {provider}: {str(e)}")
        return jsonify({'models': [], 'error': f'Failed to connect to {provider}'})

@app.route('/models/<provider>', methods=['GET'])
@with_error_handling
def get_models_by_provider(provider):
    translator = BookTranslator(provider=provider)
    try:
        available_models = translator.get_available_models()
        models = []
        for model_name in available_models:
            models.append({
                'name': model_name,
                'size': 'Unknown',
                'modified': 'Unknown'
            })
        return jsonify({'models': models})
    except Exception as e:
        logger.app_logger.error(f"Error fetching models for {provider}: {str(e)}")
        return jsonify({'models': [], 'error': f'Failed to connect to {provider}'})

@app.route('/translations', methods=['GET'])
@with_error_handling
def get_translations():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute('''
            SELECT id, filename, source_lang, target_lang, model,
                   status, progress, detected_language, created_at, 
                   updated_at, error_message
            FROM translations
            ORDER BY created_at DESC
        ''')
        translations = [dict(row) for row in cur.fetchall()]
    return jsonify({'translations': translations})

@app.route('/translations/<int:translation_id>', methods=['GET'])
@with_error_handling
def get_translation(translation_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute('SELECT * FROM translations WHERE id = ?', (translation_id,))
        translation = cur.fetchone()
        if translation:
            return jsonify(dict(translation))
        return jsonify({'error': 'Translation not found'}), 404

@app.route('/translate', methods=['POST'])
@with_error_handling
def translate():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    try:
        file = request.files['file']
        source_lang = request.form.get('sourceLanguage')
        target_lang = request.form.get('targetLanguage')
        model_name = request.form.get('model')
        provider = request.form.get('provider', 'ollama')
        use_cache = request.form.get('useCache', 'true').lower() == 'true'
        translation_mode = request.form.get('translationMode', 'two-stage')
        
        # Allow empty model_name for Google Translate only
        if not all([file, source_lang, target_lang]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Determine translation mode
        if translation_mode == 'google-only':
            skip_llm_refinement = True
            skip_google_translate = False
            model_name = 'google_translate_only'  # Set a placeholder for database
        elif translation_mode == 'llm-only':
            skip_llm_refinement = False
            skip_google_translate = True
            if not model_name or model_name.strip() == '':
                return jsonify({'error': 'Model is required for LLM-only translation mode'}), 400
        else:  # two-stage (default)
            skip_llm_refinement = False
            skip_google_translate = False
            if not model_name or model_name.strip() == '':
                skip_llm_refinement = True
                model_name = 'google_translate_only'
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='cp1251') as f:
                text = f.read()
        
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute('''
                INSERT INTO translations (
                    filename, source_lang, target_lang, model,
                    status, original_text, genre  -- Included genre
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (filename, source_lang, target_lang, model_name, 
                  'in_progress', text, 'unknown'))  # Set genre to 'unknown'
            translation_id = cur.lastrowid
        
        translator = BookTranslator(model_name=model_name, provider=provider)
        
        def generate():
            try:
                for update in translator.translate_text(text, source_lang, target_lang, translation_id, skip_llm_refinement, use_cache, skip_google_translate):
                    yield f"data: {json.dumps(update, ensure_ascii=False)}\n\n"
            except Exception as e:
                error_message = str(e)
                logger.translation_logger.error(f"Translation error: {error_message}")
                logger.translation_logger.error(traceback.format_exc())
                yield f"data: {json.dumps({'error': error_message})}\n\n"
                
        return Response(generate(), mimetype='text/event-stream')
    
    except Exception as e:
        logger.app_logger.error(f"Translation request error: {str(e)}")
        logger.app_logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            if 'filepath' in locals():
                os.remove(filepath)
        except Exception as e:
            logger.app_logger.error(f"Failed to cleanup uploaded file: {str(e)}")

@app.route('/download/<int:translation_id>', methods=['GET'])
@with_error_handling
def download_translation(translation_id):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute('''
            SELECT filename, translated_text
            FROM translations
            WHERE id = ? AND status = 'completed'
        ''', (translation_id,))
        result = cur.fetchone()
        
        if not result:
            return jsonify({'error': 'Translation not found or not completed'}), 404
        
        filename, translated_text = result
        
        # Create download file with raw text
        download_path = os.path.join(TRANSLATIONS_FOLDER, f'translated_{filename}')
        with open(download_path, 'w', encoding='utf-8') as f:
            f.write(translated_text)
            
        return send_file(
            download_path,
            as_attachment=True,
            download_name=f'translated_{filename}'
        )

@app.route('/failed-translations', methods=['GET'])
@with_error_handling
def get_failed_translations():
    return jsonify(recovery.get_failed_translations())

@app.route('/retry-translation/<int:translation_id>', methods=['POST'])
@with_error_handling
def retry_failed_translation(translation_id):
    recovery.retry_translation(translation_id)
    return jsonify({'status': 'success'})

@app.route('/metrics', methods=['GET'])
def get_metrics():
    return jsonify(monitor.get_metrics())

@app.route('/health', methods=['GET'])
def health_check():
    try:
        # Check Ollama
        ollama_status = 'disconnected'
        try:
            response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            response.raise_for_status()
            ollama_status = 'connected'
        except:
            pass
        
        # Check LMStudio
        lmstudio_status = 'disconnected'
        try:
            response = requests.get(f"{LMSTUDIO_URL}/v1/models", timeout=5)
            response.raise_for_status()
            lmstudio_status = 'connected'
        except:
            pass
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('SELECT 1')
            
        disk_usage = psutil.disk_usage('/')
        if disk_usage.percent > 90:
            logger.app_logger.warning("Low disk space")
            
        return jsonify({
            'status': 'healthy',
            'ollama': ollama_status,
            'lmstudio': lmstudio_status,
            'database': 'connected',
            'disk_usage': f"{disk_usage.percent}%"
        })
    except Exception as e:
        logger.app_logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503

def cleanup_old_data():
    while True:
        try:
            logger.app_logger.info("Running cleanup task")
            try:
                cache.cleanup_old_entries()
                logger.app_logger.info("Cache cleanup completed")
            except Exception as e:
                logger.app_logger.error(f"Cache cleanup error: {str(e)}")

            try:
                recovery.cleanup_failed_translations()
                logger.app_logger.info("Failed translations cleanup completed")
            except Exception as e:
                logger.app_logger.error(f"Failed translations cleanup error: {str(e)}")

            time.sleep(24 * 60 * 60)  # Run daily
        except Exception as e:
            logger.app_logger.error(f"Cleanup task error: {str(e)}")
            time.sleep(60 * 60)  # Retry in an hour

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_data, daemon=True)
cleanup_thread.start()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
    
