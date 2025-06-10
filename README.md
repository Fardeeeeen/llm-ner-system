# LLM NER System

A web-based system that combines **Named Entity Recognition (NER)** using spaCy with **Local LLM** capabilities via Ollama.

## Features

- **Named Entity Recognition**: Detects entities like persons, organizations, locations, etc.
- **Local LLM Integration**: Processes prompts using Ollama's REST API
- **Real-time Web Interface**: Modern, responsive UI for easy interaction
- **Console Logging**: Detailed output for debugging and monitoring
- **Error Handling**: Robust error handling and user feedback

## Prerequisites

1. **Python 3.8+**
2. **Ollama** installed and running
3. **spaCy English model**

## Installation

### 1. Clone/Download the Project
```bash
git clone <your-repo-url>
cd llm-ner-system
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 3. Install and Setup Ollama
```bash
# Install Ollama from https://ollama.ai
# After installation, pull a model:
ollama pull llama2

# Start Ollama service (if not auto-started)
ollama serve
```

## Running the Application

### 1. Start the Backend Server
```bash
python main.py
```

The server will start at `http://localhost:8000`

### 2. Access the Web Interface
Open your browser and navigate to: `http://localhost:8000`

## Usage

1. **Enter a Prompt**: Type your text in the input area (e.g., "John Doe lives in New York and works at Apple Inc.")
2. **Click Send**: The system will process your prompt
3. **View Results**: 
   - Named entities appear in the left panel
   - LLM response appears in the right panel
   - Console shows detailed logging

## API Endpoints

### `POST /process`
Process a prompt with NER and LLM
```json
{
  "prompt": "Your text here",
  "model": "llama2"
}
```

### `GET /health`
Health check endpoint

### `GET /models`
Get available Ollama models

## Example Output

### Console Output:
```
🔄 Processing prompt: 'John Doe lives in New York and works at Apple Inc.'
📝 Detected named entities:
  - John Doe (PERSON) - Confidence: 0.9
  - New York (GPE) - Confidence: 0.9
  - Apple Inc. (ORG) - Confidence: 0.9
🤖 Sending prompt to Ollama (llama2)...
💬 LLM Response: John Doe is a fictional character who...
✅ Processing completed successfully
```

### Web Interface:
- **Named Entities Panel**: Shows detected entities with their types
- **LLM Response Panel**: Displays the AI-generated response
- **Status Messages**: Success/error notifications

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   External      │
│   (HTML/JS)     │────│   Backend       │────│   Services      │
│                 │    │                 │    │                 │
│ • User Input    │    │ • NER (spaCy)   │    │ • Ollama API    │
│ • Results UI    │    │ • API Routes    │    │ • LLM Models    │
│ • Error Handling│    │ • Error Handling│    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Troubleshooting

### Common Issues:

1. **spaCy model not found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **Ollama connection error**
   - Ensure Ollama is running: `ollama serve`
   - Check if model is available: `ollama list`

3. **Port already in use**
   - Change port in `main.py`: `uvicorn.run(app, host="0.0.0.0", port=8001)`

## Project Structure
```
llm-ner-system/
├── main.py              # FastAPI backend server
├── static/
│   └── index.html      # Frontend interface
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── screenshots/       # Demo screenshots
```

## Technical Details

- **Backend**: FastAPI with async support
- **NER**: spaCy `en_core_web_sm` model
- **LLM**: Ollama REST API integration
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Styling**: Modern dark theme with animations

## Demo
Here are some screenshots demonstrating the system's functionality:

### Screenshot 1: Main Interface
![Main Interface](screenshots/Screenshot%20(32).png)

### Screenshot 2: Entity Detection
![Entity Detection](screenshots/Screenshot%20(33).png)

### Screenshot 3: Processing Results
![Processing Results](screenshots/Screenshot%20(34).png)

![Demo Video](screenshots/PrivChat.mp4)


## Future Enhancements

- [ ] Support for multiple LLM models
- [ ] Entity confidence scoring
- [ ] Batch processing
- [ ] Export functionality
- [ ] User authentication
- [ ] Chat history persistence

## License

This project is created for educational purposes as part of a coding challenge.