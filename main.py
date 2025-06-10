from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import spacy
import requests
import time
from typing import List, Dict
from tenacity import retry, stop_after_attempt, wait_fixed
from textblob import TextBlob
from gensim import corpora
from gensim.models import LdaModel
from transformers import pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="LLM NER System", version="1.0.0")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (for serving HTML)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("✅ spaCy model loaded successfully")
except IOError:
    logger.error("❌ spaCy model not found. Please run: python -m spacy download en_core_web_sm")
    exit(1)

# Load Hugging Face summarization pipeline
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    logger.info("✅ Hugging Face summarization model loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load summarization model: {str(e)}")
    summarizer = None

# Pydantic models for request/response
class PromptRequest(BaseModel):
    prompt: str
    model: str = "llama2"

class EntityRequest(BaseModel):
    text: str

class EntityResponse(BaseModel):
    text: str
    label: str
    start: int
    end: int
    confidence: float

class ProcessResponse(BaseModel):
    original_prompt: str
    entities: List[EntityResponse]
    llm_response: str
    sentiment: str
    topics: List[str]
    summary: str
    success: bool
    error: str = None

# Ollama API configuration
OLLAMA_BASE_URL = "http://localhost:11434"

def extract_entities(text: str) -> List[Dict]:
    """Extract named entities using spaCy NER"""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
            "confidence": 1.0  # spaCy default
        })
    return entities

def analyze_sentiment(text: str) -> str:
    """Analyze sentiment using TextBlob"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

def extract_topics(text: str, num_topics: int = 2) -> List[str]:
    """Extract topics using Gensim LDA"""
    try:
        doc = nlp(text)
        tokens = [token.text.lower() for token in doc if not token.is_stop and token.is_alpha]
        if len(tokens) < 5:  # Require minimum tokens for meaningful topics
            logger.warning("Insufficient tokens for topic modeling")
            return ["Insufficient text for topic modeling"]
        
        dictionary = corpora.Dictionary([tokens])
        corpus = [dictionary.doc2bow(tokens)]
        if not corpus[0]:  # Check if corpus is empty
            logger.warning("Empty corpus after processing tokens")
            return ["No topics detected"]
        
        # Reduce num_topics for short texts
        effective_num_topics = min(num_topics, len(tokens) // 2) or 1
        lda_model = LdaModel(
            corpus,
            num_topics=effective_num_topics,
            id2word=dictionary,
            passes=10,
            random_state=42,
            minimum_probability=0.0
        )
        topics = []
        for topic_id in range(effective_num_topics):
            words = [word for word, _ in lda_model.show_topic(topic_id, topn=5)]
            topics.append(", ".join(words) if words else "No topic words")
        logger.info(f"Extracted topics: {topics}")
        return topics if topics else ["No topics detected"]
    except Exception as e:
        logger.error(f"Error extracting topics: {str(e)}")
        return ["Topic modeling failed"]

def summarize_text(text: str) -> str:
    """Summarize text using Hugging Face Transformers"""
    if not summarizer:
        logger.warning("Summarizer not available")
        return "Summarization not available"
    if len(text) < 100:  # Skip for short texts
        logger.info("Text too short for summarization")
        return "Text is too short for summarization"
    try:
        summary = summarizer(text, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
        logger.info(f"Generated summary: {summary}")
        return summary
    except Exception as e:
        logger.error(f"Error summarizing text: {str(e)}")
        return "Summary generation failed"

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_ollama_api(prompt: str, model: str = "llama2") -> str:
    """Call Ollama REST API to get LLM response"""
    try:
        # Check if Ollama service is reachable
        test_response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        test_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama service check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Ollama service is not running")

    try:
        url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "max_tokens": 200
            }
        }
        logger.info(f"Calling Ollama API at {url} with model {model}")
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=60)
        logger.info(f"Ollama API call took {time.time() - start_time:.2f} seconds")
        
        if response.status_code == 500:
            logger.error(f"Ollama API returned 500 error: {response.text}")
            raise HTTPException(status_code=500, detail=f"Ollama API internal error: {response.text}")
        
        response.raise_for_status()
        result = response.json()
        llm_response = result.get("response", "No response received")
        logger.info(f"Ollama response: {llm_response[:100]}...")
        return llm_response
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error: Ollama service is not reachable - {str(e)}")
        raise HTTPException(status_code=503, detail="Ollama service is not running")
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout occurred while calling Ollama API after 60 seconds - {str(e)}")
        raise HTTPException(status_code=504, detail="Ollama request timed out after 60 seconds")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calling Ollama API: {str(e)}")

@app.get("/")
async def serve_index():
    """Serve the main HTML page"""
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "LLM NER System is running"}

@app.post("/entities", response_model=List[EntityResponse])
async def get_entities(request: EntityRequest):
    """Extract entities for real-time feedback"""
    try:
        entities = extract_entities(request.text)
        logger.info(f"Extracted entities: {[entity['text'] for entity in entities]}")
        return [EntityResponse(**entity) for entity in entities]
    except Exception as e:
        logger.error(f"Error extracting entities: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting entities: {str(e)}")

@app.post("/process", response_model=ProcessResponse)
async def process_prompt(request: PromptRequest):
    """
    Main endpoint to process user prompt:
    1. Extract named entities using spaCy
    2. Analyze sentiment, topics, and summary
    3. Attempt to send prompt to Ollama LLM (with fallback if it fails)
    4. Return combined results
    """
    try:
        logger.info(f"\nProcessing prompt: '{request.prompt}'")
        
        # Step 1: Extract named entities
        start_time = time.time()
        entities = extract_entities(request.prompt)
        logger.info(f"Detected named entities (took {time.time() - start_time:.2f} seconds): {[entity['text'] for entity in entities]}")
        
        # Step 2: Analyze sentiment
        sentiment = analyze_sentiment(request.prompt)
        logger.info(f"Sentiment: {sentiment}")
        
        # Step 3: Extract topics
        topics = extract_topics(request.prompt)
        logger.info(f"Topics: {topics}")
        
        # Step 4: Summarize text
        summary = summarize_text(request.prompt)
        logger.info(f"Summary: {summary}")
        
        # Step 5: Attempt to call Ollama LLM with a fallback
        try:
            logger.info(f"Sending prompt to Ollama ({request.model})...")
            llm_response = call_ollama_api(request.prompt, request.model)
        except Exception as e:
            llm_response = f"LLM processing failed: {str(e)}. Please check Ollama service."
            logger.error(f"Ollama API call failed: {str(e)}")
        
        # Step 6: Return combined results
        response = ProcessResponse(
            original_prompt=request.prompt,
            entities=[EntityResponse(**entity) for entity in entities],
            llm_response=llm_response,
            sentiment=sentiment,
            topics=topics,
            summary=summary,
            success=True
        )
        
        logger.info("Processing completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Error processing prompt: {str(e)}")
        return ProcessResponse(
            original_prompt=request.prompt,
            entities=[],
            llm_response="",
            sentiment="Unknown",
            topics=[],
            summary="",
            success=False,
            error=str(e)
        )

@app.get("/models")
async def get_available_models():
    """Get list of available Ollama models"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        logger.info(f"Available models: {[model['name'] for model in models]}")
        return {"models": [model["name"] for model in models]}
    except Exception as e:
        logger.error(f"Error fetching models: {str(e)}")
        return {"models": ["llama2"]}

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting LLM NER System...")
    logger.info("📍 API will be available at: http://localhost:8000")
    logger.info("🌐 Web interface will be available at: http://localhost:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000)