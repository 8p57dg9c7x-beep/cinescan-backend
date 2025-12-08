from fastapi import FastAPI, APIRouter, File, UploadFile, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import os
import logging
from pathlib import Path
from pydantic import BaseModel
import base64
import requests
import time
from pymongo import MongoClient
from bson import ObjectId

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Load API keys
TMDB_API_KEY = os.environ.get('TMDB_API_KEY')
AUDD_API_KEY = os.environ.get('AUDD_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GOOGLE_VISION_API_KEY = os.environ.get('GOOGLE_VISION_API_KEY')

# Create the main app
app = FastAPI(title="CINESCAN API", version="1.0.0")

# Create API router
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB connection
MONGO_URL = os.environ.get('MONGO_URL', 'mongodb://localhost:27017/')
mongo_client = MongoClient(MONGO_URL)
db = mongo_client['app_database']
outfits_collection = db['outfits']
beauty_collection = db['beauty_looks']

logger.info(f"MongoDB connected: {MONGO_URL}")

# Pydantic Models
class AudioRecognitionRequest(BaseModel):
    audio_base64: str

class ImageRecognitionRequest(BaseModel):
    image_base64: str

class VideoRecognitionRequest(BaseModel):
    video_base64: str

class SearchRequest(BaseModel):
    query: str

# Helper Functions
def search_tmdb_movie(query: str):
    """Search for a movie in TMDB database"""
    try:
        # Clean up the query - remove newlines and limit length
        clean_query = query.replace('\n', ' ').strip()
        
        # Limit query length to avoid TMDB API errors
        if len(clean_query) > 100:
            # Try to extract key words/title
            words = clean_query.split()
            # Look for common movie title patterns or just use first few words
            clean_query = ' '.join(words[:10])
        
        url = "https://api.themoviedb.org/3/search/movie"
        params = {
            'api_key': TMDB_API_KEY,
            'query': clean_query,
            'language': 'en-US'
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('results') and len(data['results']) > 0:
            movie = data['results'][0]
            return get_movie_details(movie['id'])
        return None
    except Exception as e:
        logger.error(f"TMDB search error: {e}")
        return None

def get_movie_details(movie_id: int):
    """Get detailed movie information from TMDB including watch providers"""
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        params = {
            'api_key': TMDB_API_KEY,
            'language': 'en-US',
            'append_to_response': 'credits,watch/providers'
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"TMDB details error: {e}")
        return None

def recognize_image_with_google_vision(image_content: bytes):
    """Use Google Vision API with WEB DETECTION for movie recognition"""
    try:
        url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
        image_base64 = base64.b64encode(image_content).decode('utf-8')
        
        request_body = {
            "requests": [{
                "image": {"content": image_base64},
                "features": [
                    {"type": "WEB_DETECTION", "maxResults": 20},
                    {"type": "TEXT_DETECTION", "maxResults": 10}
                ]
            }]
        }
        
        response = requests.post(url, json=request_body, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        web_entities = []
        detected_texts = []
        best_guess_labels = []
        
        if 'responses' in result and len(result['responses']) > 0:
            response_data = result['responses'][0]
            
            # PRIORITY 1: Web entities (most accurate for movie posters)
            if 'webDetection' in response_data:
                web_data = response_data['webDetection']
                
                # Get best guess labels first
                if 'bestGuessLabels' in web_data:
                    for label in web_data['bestGuessLabels']:
                        if 'label' in label:
                            best_guess_labels.append(label['label'])
                
                # Get web entities
                if 'webEntities' in web_data:
                    for entity in web_data['webEntities']:
                        if 'description' in entity:
                            score = entity.get('score', 0)
                            web_entities.append({
                                'text': entity['description'],
                                'score': score
                            })
            
            # PRIORITY 2: Text detection (fallback)
            if 'textAnnotations' in response_data:
                for annotation in response_data['textAnnotations']:
                    detected_texts.append(annotation.get('description', ''))
        
        return {
            'web_entities': web_entities,
            'best_guess': best_guess_labels,
            'text': detected_texts
        }
    except Exception as e:
        logger.error(f"Google Vision error: {e}")
        return {'web_entities': [], 'best_guess': [], 'text': []}

def recognize_audio_with_audd(audio_base64: str):
    """Use AudD API to recognize audio"""
    try:
        if 'base64,' in audio_base64:
            audio_base64 = audio_base64.split('base64,')[1]
        
        url = "https://api.audd.io/"
        data = {
            'api_token': AUDD_API_KEY,
            'audio': audio_base64,
            'return': 'apple_music,spotify'
        }
        
        response = requests.post(url, data=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        if result.get('status') == 'success' and result.get('result'):
            song_info = result['result']
            search_query = f"{song_info.get('title', '')} {song_info.get('artist', '')}"
            return search_query
        
        return None
    except Exception as e:
        logger.error(f"AudD error: {e}")
        return None

# API Endpoints
@api_router.get("/")
async def root():
    return {
        "message": "CINESCAN API",
        "version": "1.0.0",
        "status": "running"
    }

@api_router.post("/recognize-image")
async def recognize_image(file: UploadFile = File(...)):
    """Recognize movie from an image using web detection"""
    try:
        logger.info(f"Received image: {file.filename}, content_type: {file.content_type}")
        
        # Read the uploaded file
        image_content = await file.read()
        logger.info(f"Image content size: {len(image_content)} bytes")
        
        vision_result = recognize_image_with_google_vision(image_content)
        web_entities = vision_result.get('web_entities', [])
        best_guess = vision_result.get('best_guess', [])
        detected_texts = vision_result.get('text', [])
        
        logger.info(f"Web entities: {web_entities[:5]}")
        logger.info(f"Best guess: {best_guess}")
        
        # STRATEGY 1: Try best guess labels first (most accurate for posters)
        if best_guess:
            for guess in best_guess[:3]:
                logger.info(f"Trying best guess: '{guess}'")
                movie = search_tmdb_movie(guess)
                if movie:
                    logger.info(f"✅ FOUND via best guess: '{movie.get('title')}'")
                    return {
                        "success": True,
                        "source": "Google Web Detection (Best Guess)",
                        "movie": movie
                    }
        
        # STRATEGY 2: SMART entity matching - key insight: entity name should match movie title
        if web_entities:
            movie_candidates = []
            
            # Filter out generic terms that shouldn't be searched
            generic_terms = ['video', 'film', 'movie', 'scene', 'poster', 'film poster', 'movie poster', 
                            'illustration', 'artwork', 'cinema', 'hollywood', 'actor', 'actress',
                            'director', 'crime film', 'drama', 'thriller', 'action film', 'comedy']
            
            for entity in web_entities[:25]:
                query = entity['text']
                entity_lower = query.lower().strip()
                
                # Skip generic movie-related terms
                if entity_lower in generic_terms:
                    logger.info(f"Skipping generic term: '{query}'")
                    continue
                
                logger.info(f"Checking: '{query}'")
                movie = search_tmdb_movie(query)
                
                if movie:
                    movie_title = movie.get('title', '').lower().strip()
                    
                    # CRITICAL: Check if entity name matches the movie title
                    # If entity="Inception" and movie="Inception" → REAL MATCH
                    # If entity="Leonardo DiCaprio" and movie="Leonardo" → ACTOR, NOT THE MOVIE
                    
                    # Remove common words for matching
                    entity_clean = entity_lower.replace('the ', '').replace('a ', '').strip()
                    title_clean = movie_title.replace('the ', '').replace('a ', '').strip()
                    
                    match_score = 0
                    
                    # Perfect match: entity and title are the same
                    if entity_clean == title_clean or entity_lower == movie_title:
                        match_score = 10000  # HIGHEST PRIORITY
                        logger.info(f"  ✅ PERFECT: '{query}' = '{movie.get('title')}'")
                    
                    # Very close match: one contains the other fully
                    elif entity_clean in title_clean and len(entity_clean) > 5:
                        match_score = 5000
                        logger.info(f"  ✅ STRONG: '{query}' in '{movie.get('title')}'")
                    
                    elif title_clean in entity_clean and len(title_clean) > 5:
                        match_score = 4000
                        logger.info(f"  ✅ STRONG: '{movie.get('title')}' in '{query}'")
                    
                    # Weak match - likely actor/director
                    else:
                        match_score = 1  # Very low score
                        logger.info(f"  ❌ WEAK: '{query}' → '{movie.get('title')}' (probably actor)")
                    
                    movie_candidates.append({
                        'movie': movie,
                        'query': query,
                        'match_score': match_score,
                        'entity_score': entity['score']
                    })
            
            # Sort by match_score first (perfect matches win), then entity_score
            if movie_candidates:
                movie_candidates.sort(key=lambda x: x['match_score'], reverse=True)
                best = movie_candidates[0]
                
                # Only return if match_score is high enough (avoid actor names)
                if best['match_score'] >= 4000:
                    logger.info(f"🎯 SELECTED: '{best['movie'].get('title')}' (match_score: {best['match_score']})")
                    return {
                        "success": True,
                        "source": "Web Detection",
                        "movie": best['movie']
                    }
                else:
                    logger.info(f"⚠️  Best match score too low: {best['match_score']} for '{best['query']}'")

        
        # STRATEGY 3: Fall back to text detection (old method)
        if detected_texts and len(detected_texts) > 0:
            logger.info("Falling back to text detection")
            
            # Get all text, extract words
            all_text = detected_texts[0] if detected_texts else ""
            words = all_text.replace('\n', ' ').split()
            
            # Skip common non-movie words
            skip_words = {'the', 'a', 'and', 'of', 'in', 'to', 'for', 'starring', 'presents', 'from', 'directed', 'by', 'pictures', 'films', 'entertainment', 'studios', 'production'}
            
            # Try 2-3 word combinations
            for i in range(min(20, len(words) - 1)):
                if words[i].lower() not in skip_words:
                    # Try 2-word combo
                    query = f"{words[i]} {words[i+1]}"
                    movie = search_tmdb_movie(query)
                    if movie:
                        logger.info(f"✅ FOUND via text: '{movie.get('title')}'")
                        return {
                            "success": True,
                            "source": "Text Detection",
                            "movie": movie
                        }
                    
                    # Try 3-word combo
                    if i < len(words) - 2:
                        query = f"{words[i]} {words[i+1]} {words[i+2]}"
                        movie = search_tmdb_movie(query)
                        if movie:
                            logger.info(f"✅ FOUND via text: '{movie.get('title')}'")
                            return {
                                "success": True,
                                "source": "Text Detection",
                                "movie": movie
                            }
        
        return {
            "success": False,
            "error": "Could not identify movie. Try a clearer poster image.",
            "movie": None
        }
        
    except Exception as e:
        logger.error(f"Image recognition error: {e}")
        return {
            "success": False,
            "error": str(e),
            "movie": None
        }

@api_router.post("/recognize-image-base64")
async def recognize_image_base64(request: Request):
    """Recognize movie from base64 image (mobile-friendly)"""
    try:
        body = await request.json()
        image_base64 = body.get('image_base64')
        
        if not image_base64:
            return {
                "success": False,
                "error": "No image data provided",
                "movie": None
            }
        
        logger.info(f"Received base64 image, length: {len(image_base64)} characters")
        
        # Decode base64 to bytes
        try:
            # Remove data URL prefix if present
            if ',' in image_base64:
                image_base64 = image_base64.split(',')[1]
            
            image_content = base64.b64decode(image_base64)
            logger.info(f"Decoded image size: {len(image_content)} bytes")
        except Exception as e:
            logger.error(f"Base64 decode error: {e}")
            return {
                "success": False,
                "error": "Invalid base64 image data",
                "movie": None
            }
        
        # Use the same recognition logic as the file upload endpoint
        vision_result = recognize_image_with_google_vision(image_content)
        web_entities = vision_result.get('web_entities', [])
        best_guess = vision_result.get('best_guess', [])
        detected_texts = vision_result.get('text', [])
        
        logger.info(f"Web entities: {web_entities[:5]}")
        logger.info(f"Best guess: {best_guess}")
        
        # STRATEGY 1: Try best guess labels first
        if best_guess:
            for guess in best_guess[:3]:
                logger.info(f"Trying best guess: '{guess}'")
                movie = search_tmdb_movie(guess)
                if movie:
                    logger.info(f"✅ FOUND via best guess: '{movie.get('title')}'")
                    return {
                        "success": True,
                        "source": "Google Web Detection (Best Guess)",
                        "movie": movie
                    }
        
        # STRATEGY 2: SMART entity matching
        if web_entities:
            movie_candidates = []
            generic_terms = ['video', 'film', 'movie', 'scene', 'poster', 'film poster', 'movie poster', 
                            'illustration', 'artwork', 'cinema', 'hollywood', 'actor', 'actress',
                            'director', 'crime film', 'drama', 'thriller', 'action film', 'comedy']
            
            for entity in web_entities[:25]:
                query = entity['text']
                entity_lower = query.lower().strip()
                
                if entity_lower in generic_terms:
                    logger.info(f"Skipping generic term: '{query}'")
                    continue
                
                logger.info(f"Checking: '{query}'")
                movie = search_tmdb_movie(query)
                
                if movie:
                    movie_title = movie.get('title', '').lower().strip()
                    entity_clean = entity_lower.replace('the ', '').replace('a ', '').strip()
                    title_clean = movie_title.replace('the ', '').replace('a ', '').strip()
                    
                    match_score = 0
                    
                    if entity_clean == title_clean or entity_lower == movie_title:
                        match_score = 10000
                        logger.info(f"  ✅ PERFECT: '{query}' = '{movie.get('title')}'")
                    elif entity_clean in title_clean and len(entity_clean) > 5:
                        match_score = 5000
                        logger.info(f"  ✅ STRONG: '{query}' in '{movie.get('title')}'")
                    elif title_clean in entity_clean and len(title_clean) > 5:
                        match_score = 4000
                        logger.info(f"  ✅ STRONG: '{movie.get('title')}' in '{query}'")
                    else:
                        match_score = 100
                        logger.info(f"  ⚠️ WEAK: '{query}' -> '{movie.get('title')}'")
                    
                    movie_candidates.append({
                        'movie': movie,
                        'score': match_score,
                        'query': query
                    })
            
            if movie_candidates:
                movie_candidates.sort(key=lambda x: x['score'], reverse=True)
                best_match = movie_candidates[0]
                
                if best_match['score'] >= 4000:
                    logger.info(f"✅ BEST MATCH: '{best_match['query']}' -> '{best_match['movie'].get('title')}' (score: {best_match['score']})")
                    return {
                        "success": True,
                        "source": "Google Web Detection (Entity Match)",
                        "movie": best_match['movie']
                    }
        
        # STRATEGY 3: Text detection fallback
        if detected_texts:
            words = detected_texts[0].split()
            
            for i in range(len(words)):
                if i < len(words) - 1:
                    query = f"{words[i]} {words[i+1]}"
                    movie = search_tmdb_movie(query)
                    if movie:
                        logger.info(f"✅ FOUND via text: '{movie.get('title')}'")
                        return {
                            "success": True,
                            "source": "Text Detection",
                            "movie": movie
                        }
        
        return {
            "success": False,
            "error": "Could not identify movie. Try a clearer poster image.",
            "movie": None
        }
        
    except Exception as e:
        logger.error(f"Base64 image recognition error: {e}")
        return {
            "success": False,
            "error": str(e),
            "movie": None
        }

@api_router.post("/recognize-music-base64")
async def recognize_music_base64(request: dict):
    """Recognize music from base64 audio (mobile-friendly)"""
    try:
        audio_base64 = request.get('audio_base64')
        if not audio_base64:
            return {
                "success": False,
                "error": "No audio data provided"
            }
        
        logger.info(f"Received base64 audio, length: {len(audio_base64)}")
        
        # Use AudD to identify the song (including lyrics)
        try:
            audd_url = "https://api.audd.io/"
            audd_data = {
                'api_token': AUDD_API_KEY,
                'audio': audio_base64,
                'return': 'apple_music,spotify,lyrics'
            }
            
            response = requests.post(audd_url, data=audd_data, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"AudD response: {result}")
            
            if result.get('status') == 'success' and result.get('result'):
                song_data = result['result']
                logger.info(f"✅ Found song: {song_data.get('title')} by {song_data.get('artist')}")
                
                return {
                    "success": True,
                    "source": "AudD Music Recognition",
                    "song": {
                        "title": song_data.get('title'),
                        "artist": song_data.get('artist'),
                        "album": song_data.get('album'),
                        "release_date": song_data.get('release_date'),
                        "label": song_data.get('label'),
                        "spotify": song_data.get('spotify', {}),
                        "apple_music": song_data.get('apple_music', {}),
                        "lyrics": song_data.get('lyrics', {}),
                    }
                }
            else:
                logger.info("Song not found in AudD database")
                return {
                    "success": False,
                    "error": "Song not found. Try again with clearer audio.",
                    "song": None
                }
                
        except Exception as e:
            logger.error(f"AudD API error: {e}")
            return {
                "success": False,
                "error": "Failed to identify song",
                "song": None
            }
        
    except Exception as e:
        logger.error(f"Music recognition error: {e}")
        return {
            "success": False,
            "error": str(e),
            "song": None
        }

@api_router.post("/recognize-music")
async def recognize_music(file: UploadFile = File(...)):
    """Recognize any song/music using AudD (Shazam-like)"""
    try:
        logger.info(f"Received music: {file.filename}, content_type: {file.content_type}")
        
        # Read audio file and convert to base64
        audio_content = await file.read()
        audio_base64 = base64.b64encode(audio_content).decode('utf-8')
        
        # Use AudD to identify the song
        logger.info("🎵 Identifying song with AudD...")
        try:
            audd_url = "https://api.audd.io/"
            audd_data = {
                'api_token': AUDD_API_KEY,
                'audio': audio_base64,
                'return': 'apple_music,spotify,lyrics'
            }
            
            response = requests.post(audd_url, data=audd_data, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            if result.get('status') == 'success' and result.get('result'):
                song_data = result['result']
                logger.info(f"✅ Found song: {song_data.get('title')} by {song_data.get('artist')}")
                
                return {
                    "success": True,
                    "source": "AudD Music Recognition",
                    "song": {
                        "title": song_data.get('title'),
                        "artist": song_data.get('artist'),
                        "album": song_data.get('album'),
                        "release_date": song_data.get('release_date'),
                        "label": song_data.get('label'),
                        "spotify": song_data.get('spotify', {}),
                        "apple_music": song_data.get('apple_music', {}),
                        "lyrics": song_data.get('lyrics', {}),
                    }
                }
            else:
                logger.info("Song not found in AudD database")
                return {
                    "success": False,
                    "error": "Song not found. Try a clearer recording.",
                    "song": None
                }
                
        except Exception as e:
            logger.error(f"AudD API error: {e}")
            return {
                "success": False,
                "error": "Failed to identify song",
                "song": None
            }
        
    except Exception as e:
        logger.error(f"Music recognition error: {e}")
        return {
            "success": False,
            "error": str(e),
            "song": None
        }

@api_router.post("/recognize-audio")
async def recognize_audio(file: UploadFile = File(...)):
    """Recognize movie from audio - tries soundtrack AND dialogue recognition"""
    try:
        logger.info(f"Received audio: {file.filename}, content_type: {file.content_type}")
        
        # Read audio file and convert to base64
        audio_content = await file.read()
        audio_base64 = base64.b64encode(audio_content).decode('utf-8')
        
        # METHOD 1: Try AudD for soundtrack/music recognition
        logger.info("🎵 Trying soundtrack recognition with AudD...")
        search_query = recognize_audio_with_audd(audio_base64)
        
        if search_query:
            logger.info(f"AudD found: {search_query}")
            movie = search_tmdb_movie(search_query)
            if movie:
                logger.info(f"✅ Found movie from soundtrack: {movie.get('title')}")
                return {
                    "success": True,
                    "source": "Audio Recognition (Soundtrack)",
                    "movie": movie
                }
        
        # METHOD 2: Try dialogue recognition with OpenAI Whisper
        logger.info("🎭 Trying dialogue recognition with Whisper...")
        try:
            # Save audio temporarily
            temp_audio_path = f"/tmp/temp_audio_{int(time.time())}.mp3"
            with open(temp_audio_path, 'wb') as f:
                f.write(base64.b64decode(audio_base64))
            
            # Use OpenAI Whisper to transcribe
            if OPENAI_API_KEY:
                with open(temp_audio_path, 'rb') as f:
                    whisper_response = requests.post(
                        'https://api.openai.com/v1/audio/transcriptions',
                        headers={'Authorization': f'Bearer {OPENAI_API_KEY}'},
                        files={'file': f},
                        data={'model': 'whisper-1'},
                        timeout=30
                    )
                
                if whisper_response.status_code == 200:
                    transcription = whisper_response.json().get('text', '')
                    logger.info(f"Transcribed: {transcription[:100]}...")
                    
                    if transcription and len(transcription) > 10:
                        # Try searching TMDB with the dialogue/transcription
                        # Look for famous quotes or movie titles in the text
                        words = transcription.split()
                        
                        # Try different combinations
                        for i in range(min(len(words), 10)):
                            for length in [5, 4, 3, 2]:
                                if i + length <= len(words):
                                    query = ' '.join(words[i:i+length])
                                    movie = search_tmdb_movie(query)
                                    if movie:
                                        logger.info(f"✅ Found movie from dialogue: {movie.get('title')}")
                                        import os
                                        os.remove(temp_audio_path)
                                        return {
                                            "success": True,
                                            "source": "Audio Recognition (Dialogue)",
                                            "movie": movie,
                                            "note": "Dialogue recognition is experimental and may not be accurate"
                                        }
            
            import os
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                
        except Exception as e:
            logger.error(f"Dialogue recognition error: {e}")
        
        return {
            "success": False,
            "error": "Could not recognize audio (tried both soundtrack and dialogue)",
            "movie": None
        }
        
    except Exception as e:
        logger.error(f"Audio recognition error: {e}")
        return {
            "success": False,
            "error": str(e),
            "movie": None
        }

@api_router.post("/recognize-video")
async def recognize_video(file: UploadFile = File(...)):
    """Recognize movie from video using BOTH visual AND audio recognition"""
    import subprocess
    try:
        logger.info(f"Received video: {file.filename}, content_type: {file.content_type}")
        
        # Read video file
        video_content = await file.read()
        logger.info(f"Video content size: {len(video_content)} bytes")
        
        # Save video temporarily
        temp_video_path = f"/tmp/temp_video_{int(time.time())}.mp4"
        temp_frame_path = f"/tmp/temp_frame_{int(time.time())}.jpg"
        temp_audio_path = f"/tmp/temp_audio_{int(time.time())}.mp3"
        
        try:
            with open(temp_video_path, 'wb') as f:
                f.write(video_content)
            
            # METHOD 1: Extract frame for visual recognition
            logger.info("🎬 Attempting visual recognition from video frame...")
            result = subprocess.run([
                'ffmpeg', '-ss', '00:00:01', '-i', temp_video_path,
                '-frames:v', '1',
                temp_frame_path, '-y'
            ], capture_output=True, timeout=30)
            
            visual_movie = None
            if result.returncode == 0:
                with open(temp_frame_path, 'rb') as f:
                    frame_content = f.read()
                
                logger.info(f"Extracted frame size: {len(frame_content)} bytes")
                vision_result = recognize_image_with_google_vision(frame_content)
                web_entities = vision_result.get('web_entities', [])
                best_guess = vision_result.get('best_guess', [])
                
                logger.info(f"Video frame best guess: {best_guess}")
                
                # Try best guess
                if best_guess:
                    for guess in best_guess[:3]:
                        movie = search_tmdb_movie(guess)
                        if movie:
                            logger.info(f"✅ VISUAL: Found '{movie.get('title')}' from frame")
                            visual_movie = movie
                            break
                
                # Try web entities if no best guess match
                if not visual_movie and web_entities:
                    # Common actor names that might appear in movie scenes
                    actor_keywords = ['will smith', 'tom hanks', 'leonardo dicaprio', 'brad pitt', 
                                     'morgan freeman', 'samuel jackson', 'denzel washington',
                                     'robert downey', 'chris evans', 'scarlett johansson']
                    
                    # Filter out generic terms
                    generic_terms = ['video', 'film', 'movie', 'scene', 'poster', 'film poster', 'movie poster',
                                    'illustration', 'artwork', 'cinema', 'hollywood', 'actor', 'actress',
                                    'director', 'crime film', 'drama', 'thriller', 'action film', 'comedy']
                    
                    # Try entities that look like movie titles first
                    for entity in web_entities[:20]:
                        query = entity['text']
                        entity_lower = query.lower().strip()
                        
                        if entity_lower in generic_terms:
                            continue
                        
                        # If it's an actor name, search for their recent movies
                        is_actor = any(actor in entity_lower for actor in actor_keywords)
                        
                        if is_actor:
                            # For actor names, search TMDB for their movies and pick most popular
                            logger.info(f"Detected actor: '{query}' - searching their movies")
                            movie = search_tmdb_movie(query + " movie")
                        else:
                            movie = search_tmdb_movie(query)
                        
                        if movie:
                            movie_title = movie.get('title', '').lower().strip()
                            entity_clean = entity_lower.replace('the ', '').replace('a ', '').strip()
                            title_clean = movie_title.replace('the ', '').replace('a ', '').strip()
                            
                            # For non-actor entities, require better match
                            if not is_actor:
                                if entity_clean == title_clean or entity_clean in title_clean:
                                    logger.info(f"✅ VISUAL: Found '{movie.get('title')}' from entity")
                                    visual_movie = movie
                                    break
                            else:
                                # For actor searches, take the result
                                logger.info(f"✅ VISUAL: Found '{movie.get('title')}' from actor")
                                visual_movie = movie
                                break
            
            # METHOD 2: Extract audio for soundtrack recognition
            logger.info("🎵 Attempting audio recognition from video soundtrack...")
            result = subprocess.run([
                'ffmpeg', '-i', temp_video_path,
                '-vn', '-acodec', 'mp3', '-ar', '44100', '-ac', '2',
                '-b:a', '128k', temp_audio_path, '-y'
            ], capture_output=True, timeout=30)
            
            audio_movie = None
            if result.returncode == 0:
                with open(temp_audio_path, 'rb') as f:
                    audio_content = f.read()
                
                audio_base64 = base64.b64encode(audio_content).decode('utf-8')
                search_query = recognize_audio_with_audd(audio_base64)
                
                if search_query:
                    movie = search_tmdb_movie(search_query)
                    if movie:
                        logger.info(f"✅ AUDIO: Found '{movie.get('title')}' from soundtrack")
                        audio_movie = movie
            
            # Return best result (prioritize visual over audio)
            if visual_movie:
                return {
                    "success": True,
                    "source": "Video Visual Recognition",
                    "movie": visual_movie
                }
            elif audio_movie:
                return {
                    "success": True,
                    "source": "Video Audio Recognition (Soundtrack)",
                    "movie": audio_movie
                }
            else:
                return {
                    "success": False,
                    "error": "Could not identify movie from video (tried both visual and audio)",
                    "movie": None
                }
            
        finally:
            # Cleanup temp files
            import os
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
        
    except Exception as e:
        logger.error(f"Video recognition error: {e}")
        return {
            "success": False,
            "error": str(e),
            "movie": None
        }

@api_router.post("/search")
async def search_movie(request: SearchRequest):
    """Search for a movie by name"""
    try:
        logger.info(f"Searching for: {request.query}")
        
        movie = search_tmdb_movie(request.query)
        
        if movie:
            return {
                "success": True,
                "source": "TMDB Search",
                "movie": movie
            }
        
        return {
            "success": False,
            "error": f"Could not find movie: {request.query}",
            "movie": None
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {
            "success": False,
            "error": str(e),
            "movie": None
        }

@api_router.get("/discover/trending")
async def get_trending():
    """Get trending movies"""
    try:
        url = f"https://api.themoviedb.org/3/trending/movie/week"
        params = {'api_key': TMDB_API_KEY}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Trending error: {e}")
        return {"results": []}

@api_router.get("/discover/popular")
async def get_popular():
    """Get popular movies"""
    try:
        url = f"https://api.themoviedb.org/3/movie/popular"
        params = {'api_key': TMDB_API_KEY}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Popular error: {e}")
        return {"results": []}

@api_router.get("/discover/upcoming")
async def get_upcoming():
    """Get upcoming movies"""
    try:
        url = f"https://api.themoviedb.org/3/movie/upcoming"
        params = {'api_key': TMDB_API_KEY}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Upcoming error: {e}")
        return {"results": []}

@api_router.get("/movie/{movie_id}")
async def get_movie_detail(movie_id: int):
    """Get full movie details including cast and crew"""
    try:
        details = get_movie_details(movie_id)
        return details
    except Exception as e:
        logger.error(f"Error fetching movie details for {movie_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/movie/{movie_id}/similar")
async def get_similar_movies(movie_id: int):
    """Get similar movies for a given movie ID with fallback to recommendations"""
    try:
        # Try similar movies first
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/similar"
        params = {'api_key': TMDB_API_KEY, 'language': 'en-US', 'page': 1}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Log if empty response
        if not data.get('results'):
            logger.warning(f"TMDB returned empty similar movies for movie_id: {movie_id}")
            
            # Try recommendations endpoint as fallback
            logger.info(f"Attempting recommendations fallback for movie_id: {movie_id}")
            rec_url = f"https://api.themoviedb.org/3/movie/{movie_id}/recommendations"
            rec_response = requests.get(rec_url, params=params, timeout=10)
            rec_response.raise_for_status()
            rec_data = rec_response.json()
            
            if rec_data.get('results'):
                logger.info(f"Found {len(rec_data['results'])} recommendations for movie_id: {movie_id}")
                return rec_data
            else:
                logger.warning(f"No recommendations found either for movie_id: {movie_id}")
                return {"results": [], "fallback_message": "No similar movies available. Try browsing trending movies!"}
        
        return data
    except Exception as e:
        logger.error(f"Similar movies error for movie_id {movie_id}: {e}")
        return {"results": [], "error": str(e)}

@api_router.get("/outfits/trending")
async def get_trending_outfits():
    """Get trending outfits across all categories"""
    try:
        logger.info("Fetching trending outfits")
        
        # Get random outfits from all categories
        outfits = list(outfits_collection.aggregate([
            {"$match": {"isCelebrity": False}},
            {"$sample": {"size": 10}}
        ]))
        
        # Convert ObjectId to string
        for outfit in outfits:
            outfit['id'] = str(outfit['_id'])
            del outfit['_id']
        
        logger.info(f"Found {len(outfits)} trending outfits")
        return {"outfits": outfits}
    except Exception as e:
        logger.error(f"Trending outfits error: {e}")
        return {"outfits": []}

@api_router.get("/outfits/celebrity")
async def get_celebrity_outfits():
    """Get celebrity outfits (Dress Like Your Icon)"""
    try:
        logger.info("Fetching celebrity outfits")
        
        # Get all celebrity outfits
        outfits = list(outfits_collection.find({"isCelebrity": True}))
        
        # Convert ObjectId to string
        for outfit in outfits:
            outfit['id'] = str(outfit['_id'])
            del outfit['_id']
        
        logger.info(f"Found {len(outfits)} celebrity outfits")
        return {"outfits": outfits}
    except Exception as e:
        logger.error(f"Celebrity outfits error: {e}")
        return {"outfits": []}

@api_router.get("/outfits/{category}")
async def get_outfits(category: str):
    """Get outfits by category"""
    try:
        logger.info(f"Fetching outfits for category: {category}")
        
        # Fetch from MongoDB
        outfits = list(outfits_collection.find({"category": category}))
        
        # Convert ObjectId to string for JSON serialization
        for outfit in outfits:
            outfit['id'] = str(outfit['_id'])
            del outfit['_id']
        
        logger.info(f"Found {len(outfits)} outfits for category: {category}")
        return {"outfits": outfits, "category": category}
    except Exception as e:
        logger.error(f"Outfits error: {e}")
        return {"outfits": [], "category": category}

@api_router.post("/outfits")
async def create_outfit(outfit: dict):
    """Create a new outfit (for admin use)"""
    try:
        # This endpoint will be used to add outfits to the database
        # For now, just return success
        logger.info(f"Creating outfit: {outfit.get('title', 'Unnamed')}")
        return {"success": True, "message": "Outfit endpoint ready for data"}
    except Exception as e:
        logger.error(f"Create outfit error: {e}")
        return {"success": False, "error": str(e)}

# ========== BEAUTY ENDPOINTS ==========

@api_router.get("/beauty/{category}")
async def get_beauty_looks(category: str):
    """Get beauty looks by category"""
    try:
        logger.info(f"Fetching beauty looks for category: {category}")
        
        # Get beauty looks for the specified category
        looks = list(beauty_collection.find({"category": category}))
        
        # Convert ObjectId to string
        for look in looks:
            look['id'] = str(look['_id'])
            del look['_id']
        
        logger.info(f"Found {len(looks)} beauty looks for category: {category}")
        return {"looks": looks}
    except Exception as e:
        logger.error(f"Beauty looks error: {e}")
        return {"looks": []}

@api_router.get("/beauty/trending")
async def get_trending_beauty():
    """Get trending beauty looks"""
    try:
        logger.info("Fetching trending beauty looks")
        
        # Get random trending looks
        looks = list(beauty_collection.aggregate([
            {"$sample": {"size": 10}}
        ]))
        
        # Convert ObjectId to string
        for look in looks:
            look['id'] = str(look['_id'])
            del look['_id']
        
        logger.info(f"Found {len(looks)} trending beauty looks")
        return {"looks": looks}
    except Exception as e:
        logger.error(f"Trending beauty error: {e}")
        return {"looks": []}

@api_router.post("/music/search")
async def search_music(request: Request):
    """Search for a song by title and artist using AudD API"""
    try:
        body = await request.json()
        title = body.get('title', '')
        artist = body.get('artist', '')
        
        if not title or not artist:
            return {
                "success": False,
                "error": "Title and artist are required"
            }
        
        logger.info(f"Searching for song: {title} by {artist}")
        
        # Use AudD search endpoint
        audd_url = "https://api.audd.io/findLyrics/"
        params = {
            'api_token': AUDD_API_KEY,
            'q': f"{artist} {title}",
            'return': 'apple_music,spotify,lyrics'
        }
        
        response = requests.get(audd_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'success' and data.get('result'):
            song_data = data['result']
            logger.info(f"✅ Found song: {song_data.get('title')} by {song_data.get('artist')}")
            
            return {
                "success": True,
                "source": "audd_search",
                "song": song_data
            }
        else:
            logger.warning(f"Song not found: {title} by {artist}")
            return {
                "success": False,
                "error": "Song not found in database",
                "song": None
            }
            
    except requests.exceptions.Timeout:
        logger.error("AudD search timeout")
        return {
            "success": False,
            "error": "Search request timed out",
            "song": None
        }
    except Exception as e:
        logger.error(f"Music search error: {e}")
        return {
            "success": False,
            "error": str(e),
            "song": None
        }

@api_router.get("/lyrics/{query}")
async def get_lyrics(query: str):
    """Fetch lyrics for a song using AudD API"""
    try:
        logger.info(f"Fetching lyrics for: {query}")
        
        # Use AudD findLyrics endpoint
        url = "https://api.audd.io/findLyrics/"
        params = {
            "q": query,
            "api_token": AUDD_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # If lyrics found
        if data.get("status") == "success" and data.get("result"):
            # AudD returns an array of results
            results = data["result"]
            if isinstance(results, list) and len(results) > 0:
                lyrics = results[0].get("lyrics")
                song_title = results[0].get("title", "Unknown")
                artist = results[0].get("artist", "Unknown")
                
                if lyrics:
                    logger.info(f"✅ Found lyrics for: {song_title} by {artist}")
                    return {
                        "success": True,
                        "lyrics": lyrics,
                        "title": song_title,
                        "artist": artist
                    }
        
        # No lyrics found
        logger.warning(f"No lyrics found for: {query}")
        return {
            "success": False,
            "lyrics": None,
            "message": "No lyrics found for this song yet."
        }
        
    except requests.exceptions.Timeout:
        logger.error("AudD lyrics request timeout")
        return {
            "success": False,
            "lyrics": None,
            "message": "Request timed out. Please try again."
        }
    except Exception as e:
        logger.error(f"Lyrics fetch error: {e}")
        return {
            "success": False,
            "lyrics": None,
            "message": "Unable to fetch lyrics at this time."
        }

# Include router
app.include_router(api_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
