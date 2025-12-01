"""
CINESCAN v1.0 - Clean Backend
Rebuilt from scratch per RESET v1.0 requirements
"""
from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging
import base64
import requests
import json
import subprocess
import tempfile
from pydub import AudioSegment
import shutil
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
AUDD_API_KEY = os.environ.get("AUDD_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TMDB_API_TOKEN = os.environ.get("TMDB_API_TOKEN")
GOOGLE_VISION_API_KEY = os.environ.get("GOOGLE_VISION_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# Startup verification
print("[CINESCAN v1.0] Starting backend...")
print(f"[CINESCAN v1.0] AudD API: {'✅ Configured' if AUDD_API_KEY else '❌ Missing'}")
print(f"[CINESCAN v1.0] OpenAI Whisper: {'✅ Configured' if OPENAI_API_KEY else '❌ Missing'}")
print(f"[CINESCAN v1.0] TMDB API: {'✅ Configured' if TMDB_API_TOKEN else '❌ Missing'}")
print(f"[CINESCAN v1.0] Vision API: {'✅ Configured' if GOOGLE_VISION_API_KEY else '❌ Missing'}")

# Configure audio/video processing - try multiple FFmpeg locations
ffmpeg_paths = [
    shutil.which("ffmpeg"),  # System PATH
    "/usr/bin/ffmpeg",  # Standard Linux location
    "/usr/local/bin/ffmpeg",  # Alternative location
    "/opt/homebrew/bin/ffmpeg",  # macOS Homebrew
]

ffmpeg_path = next((path for path in ffmpeg_paths if path and os.path.exists(path)), None)

if ffmpeg_path:
    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffmpeg = ffmpeg_path
    AudioSegment.ffprobe = ffmpeg_path.replace('ffmpeg', 'ffprobe')
    logger.info(f"[FFmpeg] Found at: {ffmpeg_path}")
else:
    logger.warning("[FFmpeg] NOT FOUND - Video recognition will fail!")

# FastAPI app
app = FastAPI(title="CINESCAN v1.0 API")
api_router = APIRouter(prefix="/api")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class AudioRecognitionRequest(BaseModel):
    audio_base64: str

class MovieSearchRequest(BaseModel):
    query: str

# ===== UTILITY FUNCTIONS =====

def standardize_response(success: bool, source: str, movie=None, raw=None, error=None):
    """Standardized API response format"""
    return {
        "success": success,
        "source": source,
        "movie": movie,
        "raw": raw,
        "error": error
    }

def search_tmdb(query: str):
    """Search TMDB for movie"""
    try:
        headers = {'Authorization': f'Bearer {TMDB_API_TOKEN}', 'accept': 'application/json'}
        params = {'query': query, 'include_adult': 'false', 'language': 'en-US'}
        
        response = requests.get(f"{TMDB_BASE_URL}/search/movie", params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            results = response.json().get('results', [])
            if results:
                return results[0]
        return None
    except Exception as e:
        logger.error(f"TMDB search error: {e}")
        return None

def get_movie_details(movie_id: int):
    """Get full movie details from TMDB"""
    try:
        headers = {'Authorization': f'Bearer {TMDB_API_TOKEN}', 'accept': 'application/json'}
        response = requests.get(f"{TMDB_BASE_URL}/movie/{movie_id}", headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        logger.error(f"Get movie details error: {e}")
        return None

# ===== AUDIO RECOGNITION (AudD) =====

def recognize_audio_audd(audio_data: bytes):
    """Recognize audio using AudD API with enhanced format support"""
    try:
        logger.info(f"[AudD] Processing {len(audio_data)} bytes")
        
        # Detect audio format and save with appropriate extension
        # Support: .wav, .mp3, .m4a, .webm, .mov, .aac
        audio_formats = ['.mp3', '.wav', '.m4a', '.webm', '.ogg', '.aac']
        
        input_path = None
        wav_path = None
        
        try:
            # Try different format extensions until one works
            for ext in audio_formats:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as input_file:
                        input_file.write(audio_data)
                        input_path = input_file.name
                    
                    # Try to load with pydub
                    audio = AudioSegment.from_file(input_path)
                    logger.info(f"[AudD] Successfully loaded audio as {ext}")
                    break
                except Exception as load_error:
                    if input_path and os.path.exists(input_path):
                        os.unlink(input_path)
                    input_path = None
                    logger.debug(f"[AudD] Failed to load as {ext}: {load_error}")
                    continue
            
            if not input_path or not audio:
                raise Exception("Could not load audio in any supported format")
            
            # Normalize and convert to WAV for AudD
            audio = audio.normalize()
            
            # Ensure audio is at least 1 second long
            if len(audio) < 1000:  # Less than 1 second
                logger.warning("[AudD] Audio too short, padding to 1 second")
                silence = AudioSegment.silent(duration=1000 - len(audio))
                audio = audio + silence
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as wav_file:
                audio.export(
                    wav_file.name,
                    format='wav',
                    parameters=['-ar', '44100', '-ac', '1', '-ab', '128k']
                )
                with open(wav_file.name, 'rb') as f:
                    wav_data = f.read()
                wav_path = wav_file.name
            
            logger.info(f"[AudD] Converted to WAV: {len(wav_data)} bytes, duration: {len(audio)/1000:.1f}s")
            
        finally:
            # Cleanup temp files
            if input_path and os.path.exists(input_path):
                os.unlink(input_path)
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)
        
        # Call AudD API
        files = {'file': ('audio.wav', wav_data, 'audio/wav')}
        data = {'api_token': AUDD_API_KEY, 'return': 'apple_music,spotify', 'limit': '1'}
        
        response = requests.post('https://api.audd.io/recognize', files=files, data=data, timeout=15)
        result = response.json()
        
        logger.info(f"[AudD] Response: {result.get('status')}")
        
        if result.get('status') == 'success' and result.get('result'):
            audio_result = result['result']
            title = audio_result.get('title', '')
            artist = audio_result.get('artist', '')
            
            # Search TMDB
            search_query = f"{title} {artist}" if artist else title
            movie = search_tmdb(search_query)
            
            if movie:
                full_movie = get_movie_details(movie['id'])
                return standardize_response(
                    success=True,
                    source="AudD",
                    movie=full_movie,
                    raw=audio_result,
                    error=None
                )
        
        return standardize_response(success=False, source="AudD", error="No match found")
        
    except Exception as e:
        logger.error(f"[AudD] Error: {e}")
        return standardize_response(success=False, source="AudD", error=str(e))

# ===== WHISPER TRANSCRIPTION (OpenAI) =====

def transcribe_audio_whisper(audio_data: bytes):
    """Transcribe audio using OpenAI Whisper API and intelligently match to movies"""
    try:
        logger.info(f"[Whisper] Transcribing {len(audio_data)} bytes")
        
        # Save audio to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
        
        # Call Whisper API
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        with open(temp_path, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        
        # Cleanup
        os.unlink(temp_path)
        
        logger.info(f"[Whisper] Full transcription: {transcript}")
        
        if not transcript or len(transcript) < 5:
            return standardize_response(
                success=False, 
                source="Whisper", 
                error="No speech detected in audio",
                raw={"transcript": transcript}
            )
        
        # Enhanced TMDB search with transcript
        # Try multiple search strategies
        
        # Strategy 1: Use GPT to extract movie title from dialogue
        try:
            logger.info("[Whisper] Using GPT to identify movie from dialogue...")
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a movie expert. Given dialogue from a movie, identify the movie title. Respond ONLY with the movie title, nothing else. If you cannot identify it, respond with 'UNKNOWN'."},
                    {"role": "user", "content": f"What movie is this dialogue from?\n\nDialogue: {transcript}"}
                ],
                temperature=0,
                max_tokens=50
            )
            
            movie_title = completion.choices[0].message.content.strip()
            logger.info(f"[Whisper] GPT identified movie: {movie_title}")
            
            if movie_title and movie_title != "UNKNOWN" and len(movie_title) > 2:
                movie = search_tmdb(movie_title)
                if movie:
                    full_movie = get_movie_details(movie['id'])
                    return standardize_response(
                        success=True,
                        source="Whisper + GPT",
                        movie=full_movie,
                        raw={"transcript": transcript, "gpt_title": movie_title},
                        error=None
                    )
        except Exception as gpt_error:
            logger.warning(f"[Whisper] GPT identification failed: {gpt_error}")
        
        # Strategy 2: Search TMDB with transcript keywords
        # Extract potential movie-related words
        words = transcript.split()
        search_queries = []
        
        # Try chunks of the transcript
        if len(words) > 3:
            search_queries.append(' '.join(words[:5]))  # First 5 words
            search_queries.append(' '.join(words[-5:]))  # Last 5 words
        search_queries.append(transcript[:100])  # First 100 characters
        
        for query in search_queries:
            movie = search_tmdb(query)
            if movie:
                full_movie = get_movie_details(movie['id'])
                return standardize_response(
                    success=True,
                    source="Whisper",
                    movie=full_movie,
                    raw={"transcript": transcript, "search_query": query},
                    error=None
                )
        
        # No match found
        return standardize_response(
            success=False, 
            source="Whisper", 
            error="Could not identify movie from dialogue",
            raw={"transcript": transcript}
        )
        
    except Exception as e:
        logger.error(f"[Whisper] Error: {e}")
        import traceback
        logger.error(f"[Whisper] Traceback: {traceback.format_exc()}")
        return standardize_response(success=False, source="Whisper", error=str(e))

def recognize_audio_enhanced(audio_data: bytes):
    """Enhanced audio recognition using both AudD and Whisper"""
    try:
        # Try AudD first (for music/soundtracks)
        logger.info("[Enhanced Audio] Trying AudD for music fingerprinting...")
        audd_result = recognize_audio_audd(audio_data)
        if audd_result['success']:
            return audd_result
        
        # If AudD fails, try Whisper (for dialogue/speech)
        logger.info("[Enhanced Audio] AudD no match, trying Whisper transcription...")
        whisper_result = transcribe_audio_whisper(audio_data)
        if whisper_result['success']:
            return whisper_result
        
        # Both failed
        return standardize_response(
            success=False,
            source="AudD + Whisper",
            error="No match found with either music fingerprinting or speech transcription"
        )
        
    except Exception as e:
        logger.error(f"[Enhanced Audio] Error: {e}")
        return standardize_response(success=False, source="Enhanced", error=str(e))



# ===== VIDEO PROCESSING =====

def extract_audio_from_video(video_data: bytes):
    """Extract audio from video using ffmpeg"""
    try:
        logger.info(f"[Video] Extracting audio from {len(video_data)} bytes")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as video_file:
            video_file.write(video_data)
            video_path = video_file.name
        
        audio_path = tempfile.mktemp(suffix='.wav')
        
        cmd = [
            '/usr/bin/ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            audio_path, '-y'
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        
        if result.returncode != 0:
            raise Exception(f"ffmpeg failed: {result.stderr}")
        
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        
        os.unlink(video_path)
        os.unlink(audio_path)
        
        logger.info(f"[Video] Extracted {len(audio_data)} bytes audio")
        return audio_data
        
    except Exception as e:
        logger.error(f"[Video] Extraction error: {e}")
        raise

# ===== IMAGE RECOGNITION (Google Vision) =====

def recognize_image_vision(image_data: bytes):
    """Recognize image using Google Vision API"""
    try:
        logger.info(f"[Vision] Processing {len(image_data)} bytes")
        
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        vision_url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
        
        request_body = {
            "requests": [{
                "image": {"content": image_base64},
                "features": [
                    {"type": "TEXT_DETECTION", "maxResults": 10},
                    {"type": "LABEL_DETECTION", "maxResults": 10},
                    {"type": "LOGO_DETECTION", "maxResults": 5},
                    {"type": "WEB_DETECTION", "maxResults": 10}
                ]
            }]
        }
        
        response = requests.post(vision_url, json=request_body, timeout=15)
        vision_result = response.json()
        
        logger.info(f"[Vision] Full response: {json.dumps(vision_result, indent=2)[:500]}")
        
        # Extract text
        texts = []
        labels = []
        web_entities = []
        
        if 'responses' in vision_result and vision_result['responses']:
            resp = vision_result['responses'][0]
            
            # Get text
            if 'textAnnotations' in resp:
                texts = [ann['description'] for ann in resp['textAnnotations']]
                logger.info(f"[Vision] Found texts: {texts[:3]}")
            
            # Get labels
            if 'labelAnnotations' in resp:
                labels = [ann['description'] for ann in resp['labelAnnotations']]
                logger.info(f"[Vision] Found labels: {labels}")
            
            # Get web entities (most important for movie posters!)
            if 'webDetection' in resp and 'webEntities' in resp['webDetection']:
                web_entities = [ent.get('description', '') for ent in resp['webDetection']['webEntities'] if ent.get('description')]
                logger.info(f"[Vision] Found web entities: {web_entities}")
        
        # Try searching with web entities first (most accurate for movie posters)
        # Prioritize entities that look like movie titles (not person names)
        priority_entities = []
        person_entities = []
        
        for entity in web_entities[:10]:
            # Skip generic terms
            if entity.lower() in ['poster', 'image', 'action', 'film', 'movie']:
                continue
            # Separate person names from movie titles
            if any(word in entity.lower() for word in ['poster', 'movie', 'film']):
                priority_entities.append(entity)
            elif ' ' in entity and len(entity.split()) <= 2:
                # Likely a person name
                person_entities.append(entity)
            else:
                priority_entities.append(entity)
        
        # Search priority entities first
        for entity in priority_entities[:5]:
            logger.info(f"[Vision] Searching TMDB for web entity: {entity}")
            movie = search_tmdb(entity)
            if movie:
                full_movie = get_movie_details(movie['id'])
                return standardize_response(
                    success=True,
                    source="Vision (Web)",
                    movie=full_movie,
                    raw=vision_result,
                    error=None
                )
        
        # Try searching with extracted text
        if texts:
            search_query = texts[0][:100]
            logger.info(f"[Vision] Searching TMDB for text: {search_query}")
            movie = search_tmdb(search_query)
            
            if movie:
                full_movie = get_movie_details(movie['id'])
                return standardize_response(
                    success=True,
                    source="Vision (Text)",
                    movie=full_movie,
                    raw=vision_result,
                    error=None
                )
        
        # Try labels as last resort
        for label in labels[:3]:
            if 'movie' in label.lower() or 'film' in label.lower():
                continue
            logger.info(f"[Vision] Searching TMDB for label: {label}")
            movie = search_tmdb(label)
            if movie:
                full_movie = get_movie_details(movie['id'])
                return standardize_response(
                    success=True,
                    source="Vision (Label)",
                    movie=full_movie,
                    raw=vision_result,
                    error=None
                )
        
        logger.warning(f"[Vision] No matches found. Texts: {texts[:2]}, Labels: {labels[:3]}, Entities: {web_entities[:3]}")
        return standardize_response(success=False, source="Vision", error="No movie found in image")
        
    except Exception as e:
        logger.error(f"[Vision] Error: {e}")
        return standardize_response(success=False, source="Vision", error=str(e))

# ===== API ENDPOINTS =====

@api_router.get("/")
async def root():
    return {"message": "CINESCAN v1.0 API", "status": "running"}

@api_router.post("/recognize-audio")
async def recognize_audio_endpoint(request: AudioRecognitionRequest):
    """Audio recognition endpoint (Enhanced with Whisper)"""
    try:
        audio_data = base64.b64decode(
            request.audio_base64.split(',')[1] if ',' in request.audio_base64 else request.audio_base64
        )
        return recognize_audio_enhanced(audio_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/recognize-video")
async def recognize_video_endpoint(file: UploadFile = File(...)):
    """Video recognition endpoint (Enhanced with Whisper)"""
    try:
        video_data = await file.read()
        audio_data = extract_audio_from_video(video_data)
        return recognize_audio_enhanced(audio_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/recognize-image")
async def recognize_image_endpoint(file: UploadFile = File(...)):
    """Image recognition endpoint"""
    try:
        image_data = await file.read()
        return recognize_image_vision(image_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/search")
async def search_endpoint(request: MovieSearchRequest):
    """Movie search endpoint"""
    try:
        movie = search_tmdb(request.query)
        if movie:
            full_movie = get_movie_details(movie['id'])
            return standardize_response(
                success=True,
                source="TMDB",
                movie=full_movie,
                raw=None,
                error=None
            )
        return standardize_response(success=False, source="TMDB", error="No results found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount router
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
