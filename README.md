# 🩺 Zolution — Backend API

Multi-tenant Agentic AI backend for clinical appointment scheduling and WhatsApp automation.

## 🚀 Architecture & Tech Stack

- **Framework:** FastAPI (Python 3.12, Async)
- **Database:** PostgreSQL 16 with native **Row-Level Security (RLS)** for multi-tenant isolation
- **ORM & Migrations:** SQLAlchemy (Async) & Alembic
- **LLM Multi-Provider Factory:** OpenAI (GPT-4o), Anthropic (Claude 3.5 Sonnet / Haiku with Prompt Caching), Google (Gemini 2.5 Flash)
- **Agentic AI:** Function Calling (Tool Calling) for `check_availability` and `book_appointment`
- **Integrations:**
  - Google Workspace / Google Calendar API (OAuth 2.0 flow)
  - Meta WhatsApp Cloud API (Webhook processing with `BackgroundTasks`)

## 📁 Project Structure

```text
├── alembic/                # Database migration scripts (Asyncpg)
├── app/
│   ├── admin/             # Superadmin operations (RLS bypass)
│   ├── agents/            # AI Agent prompt and model configurations
│   ├── api/v1/            # FastAPI REST endpoints
│   ├── conversations/     # WhatsApp chat message loops
│   ├── core/              # Config, DB engine, security middleware
│   ├── integrations/      # Google Calendar & WhatsApp clients
│   ├── llm_providers/     # Provider adapters & Tool schemas
│   └── tenants/           # Organization models and services
├── tests/                 # Unit and integration test suite
├── Dockerfile             # Multi-stage production container
├── render.yaml            # 1-click cloud deployment blueprint for Render
└── requirements.txt       # Project dependencies
```

## 🛠️ Local Development Setup

1. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # source .venv/bin/activate # Linux / macOS
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   copy .env.example .env
   ```

4. **Apply database migrations:**
   ```bash
   alembic upgrade head
   ```

5. **Start development server:**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
   API Docs available at: `http://localhost:8000/docs`

## ☁️ Cloud Deployment (Render)

This repository includes a `render.yaml` Blueprint. When deployed on Render:
1. Connect this repository as a **Blueprint**.
2. Render automatically provisions:
   - **`zolution-db`**: PostgreSQL 16 database.
   - **`zolution-api`**: FastAPI Web Service running migrations and Uvicorn.
