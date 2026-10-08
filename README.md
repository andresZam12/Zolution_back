# ⚙️ Zolution — Backend API

Universal Multi-Tenant Agentic AI SaaS platform for WhatsApp communication, customer engagement, and automated appointment scheduling via Google Calendar across any industry and business size.

---

## 🚀 Architecture & Tech Stack

- **Framework:** FastAPI (Python 3.12, Async)
- **Database:** PostgreSQL 16 with native **Row-Level Security (RLS)** for strict multi-tenant isolation
- **ORM & Migrations:** SQLAlchemy (Async) & Alembic
- **LLM Multi-Provider Factory:**
  - Anthropic: Claude 3.5 Sonnet / Claude Haiku 4.5 (with prompt caching)
  - OpenAI: GPT-4o
  - Google: Gemini 2.5 Flash
- **Agentic AI:** Tool Calling / Function Calling for `check_availability` and `book_appointment`
- **Integrations:**
  - Google Workspace / Google Calendar API (OAuth 2.0 flow)
  - Meta WhatsApp Cloud API (Webhook processing with `BackgroundTasks`)

---

## 📁 Project Structure

```text
├── alembic/                # Database migration scripts (Asyncpg)
├── app/
│   ├── admin/             # Superadmin operations (RLS bypass)
│   ├── agents/            # AI Agent prompt and model configurations
│   ├── api/v1/            # FastAPI REST endpoints (conversations, appointments, agents, integrations)
│   ├── conversations/     # WhatsApp chat message loops
│   ├── core/              # Config, DB engine, security middleware
│   ├── integrations/      # Google Calendar & WhatsApp clients
│   ├── llm_providers/     # Provider adapters & Tool schemas
│   └── tenants/           # Organization models and services
├── scripts/               # Seeding, diagnostics and smoke tests
├── tests/                 # Unit and integration test suite
├── Dockerfile             # Multi-stage production container
├── render.yaml            # 1-click cloud deployment blueprint for Render
└── requirements.txt       # Project dependencies
```

---

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

4. **Apply database migrations & seed demo businesses:**
   ```bash
   alembic upgrade head
   python -m scripts.seed_data
   ```

5. **Start development server:**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
   Interactive API Docs available at: `http://localhost:8000/docs`

---

## ☁️ Cloud Deployment on Render

This repository is configured with a **Render Blueprint** (`render.yaml`) for 1-click deployment:

1. Log in to [Render Dashboard](https://dashboard.render.com).
2. Click **New +** and select **Blueprint**.
3. Connect your repository: `andresZam12/Zolution_back`.
4. Render will read `render.yaml` and provision:
   - **`zolution-db`**: Managed PostgreSQL 16 database.
   - **`zolution-api`**: FastAPI Web Service running:
     ```bash
     alembic upgrade head && python -m scripts.seed_data && uvicorn app.main:app --host 0.0.0.0 --port $PORT
     ```
5. Set your optional LLM API keys in the Render environment settings:
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`
   - `GOOGLE_API_KEY`
6. Once deployed, copy your backend URL (e.g. `https://zolution-api.onrender.com`) and configure it as `NEXT_PUBLIC_API_URL` in your Vercel frontend project.
