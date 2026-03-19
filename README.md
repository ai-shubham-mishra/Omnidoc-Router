# Omnidoc Router

LLM-powered chat router for Omnidoc workflows. This service acts as the conversational interface — it identifies user intent, collects required inputs through natural language, and dispatches workflow executions to the [OmniDoc-AgenticAPI](https://github.com/ai-shubham-mishra/OmniDoc-AgenticAPI) backend.

## Architecture

```
Frontend  →  Omnidoc-Router (port 8500)  →  OmniDoc-AgenticAPI (port 8400)
              ├── Intent matching (Gemini)
              ├── Input collection
              ├── Session management (MongoDB)
              └── HITL confirmation flow
```

## Project Structure

```
├── app.py                  # FastAPI entry point & endpoints
├── core/
│   ├── orchestrator.py     # Main orchestrator (state machine)
│   ├── session_manager.py  # MongoDB chat session CRUD
│   ├── gemini_client.py    # Gemini LLM interactions
│   └── workflow_matcher.py # Intent → workflow matching
├── handlers/
│   ├── input_collector.py  # Parse & collect workflow inputs
│   ├── request_builder.py  # Build HTTP requests for AgenticAPI
│   └── file_handler.py     # File upload management
├── models/
│   └── api_contracts.py    # Pydantic request/response schemas
├── auth/
│   ├── jwt_auth.py         # JWT verification
│   ├── models.py           # UserContext model
│   ├── dependencies.py     # FastAPI auth dependencies
│   └── middleware.py       # Global JWT middleware
├── utils/
│   └── config.py           # App version config
├── Dockerfile
├── docker-compose.yml
├── docker-compose.dev.yml
└── requirements.txt
```

## Setup

### Local Development

```bash
# 1. Clone the repo
git clone https://github.com/ai-shubham-mishra/Omnidoc-Router.git
cd Omnidoc-Router

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your actual values

# 5. Run
uvicorn app:app --host 0.0.0.0 --port 8500 --reload
```

### Docker

```bash
# Production
docker compose up -d --build

# Development (port 8501)
docker compose -f docker-compose.dev.yml up -d --build
```

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `MONGO_DB_URL` | MongoDB connection string | Yes |
| `MONGO_DB_NAME` | Database name | Yes |
| `REGISTERED_WORKFLOW_COLLECTION_NAME` | Collection for registered workflows | Yes |
| `GOOGLE_API_KEY` | Google Gemini API key | Yes |
| `JWT_SECRET` | JWT secret (must match AgenticAPI) | Yes |
| `AGENTICAPI_BASE_URL` | URL of AgenticAPI service | Yes |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/router/chat` | Chat with the router |
| `POST` | `/api/router/upload` | Upload files for a session |
| `POST` | `/api/router/confirm` | Confirm/cancel a HITL step |
| `GET` | `/api/router/session/{id}` | Get session details |
| `DELETE` | `/api/router/session/{id}` | Delete a session |
| `GET` | `/health` | Health check |

## Conversation Flow

1. **User sends message** → Router identifies the workflow via keyword/Gemini matching
2. **Input collection** → Router asks for missing inputs conversationally
3. **Execution** → Router calls AgenticAPI endpoint with collected data
4. **HITL** → If workflow pauses for confirmation, Router presents data to user
5. **Completion** → Router formats and returns the final result
