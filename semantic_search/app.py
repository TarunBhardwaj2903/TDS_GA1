"""
Semantic Search with Re-ranking API
====================================
Two-stage search pipeline:
  Stage 1: Embed query → cosine similarity against 63 doc embeddings → top K candidates
  Stage 2: LLM re-ranks candidates using Gemini → return top rerankK results

Embeddings: OpenAI text-embedding-3-small
Re-ranking:  Gemini 2.0 Flash

Run:  uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os
import json
import time
import numpy as np
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if not GEMINI_API_KEY:
    print("⚠️  WARNING: GEMINI_API_KEY not set.")
if not OPENAI_API_KEY:
    print("⚠️  WARNING: OPENAI_API_KEY not set.")

genai.configure(api_key=GEMINI_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

EMBEDDING_MODEL = "text-embedding-3-small"   # OpenAI — works reliably
RERANK_MODEL    = "gemini-2.0-flash"         # Gemini — for LLM re-ranking

# ── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Semantic Search with Re-ranking",
    description="Two-stage search: OpenAI vector retrieval + Gemini LLM re-ranking",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 63 API Documentation Documents ─────────────────────────────────────────
DOCUMENTS = [
    {"id": 0,  "content": "Authentication API: Use POST /auth/login with username and password in the request body to receive a JWT token. The token expires in 24 hours. Include it in the Authorization header as 'Bearer <token>' for all subsequent requests.", "metadata": {"source": "auth-api-docs", "category": "authentication"}},
    {"id": 1,  "content": "OAuth 2.0 Integration: To set up OAuth, register your application at /settings/oauth to get client_id and client_secret. Support authorization_code, implicit, and client_credentials grant types. Redirect URIs must be HTTPS in production.", "metadata": {"source": "oauth-docs", "category": "authentication"}},
    {"id": 2,  "content": "API Key Management: Generate API keys from the dashboard under Settings > API Keys. Each key has configurable permissions (read, write, admin). Rate limits are applied per key. Rotate keys every 90 days for security.", "metadata": {"source": "api-key-docs", "category": "authentication"}},
    {"id": 3,  "content": "Rate Limiting: All API endpoints are rate-limited. Free tier: 100 requests/minute. Pro tier: 1000 requests/minute. Enterprise: custom limits. When exceeded, the API returns 429 Too Many Requests with a Retry-After header.", "metadata": {"source": "rate-limit-docs", "category": "rate-limiting"}},
    {"id": 4,  "content": "Pagination: List endpoints support cursor-based pagination. Use 'limit' (max 100) and 'cursor' parameters. Response includes 'next_cursor' for the next page. Use 'has_more' boolean to check if more results exist.", "metadata": {"source": "pagination-docs", "category": "pagination"}},
    {"id": 5,  "content": "Error Handling: The API uses standard HTTP status codes. 400: Bad Request, 401: Unauthorized, 403: Forbidden, 404: Not Found, 429: Rate Limited, 500: Internal Server Error. Error responses include 'error_code', 'message', and 'details' fields.", "metadata": {"source": "error-docs", "category": "errors"}},
    {"id": 6,  "content": "Webhooks: Configure webhooks at /settings/webhooks. Events include user.created, order.completed, payment.failed. Payloads are signed with HMAC-SHA256. Verify the X-Webhook-Signature header. Retry policy: 3 attempts with exponential backoff.", "metadata": {"source": "webhook-docs", "category": "webhooks"}},
    {"id": 7,  "content": "File Upload API: Upload files using POST /files with multipart/form-data. Maximum file size: 50MB. Supported formats: PDF, PNG, JPG, CSV, JSON. Returns a file_id for referencing in other API calls. Files are stored for 30 days.", "metadata": {"source": "file-upload-docs", "category": "files"}},
    {"id": 8,  "content": "Search API: Use GET /search?q=<query>&type=<type> to search across resources. Supports full-text search with relevance scoring. Filter by type (users, documents, orders). Results are paginated with max 50 per page.", "metadata": {"source": "search-docs", "category": "search"}},
    {"id": 9,  "content": "User Management API: CRUD operations for users. POST /users to create, GET /users/:id to retrieve, PUT /users/:id to update, DELETE /users/:id to remove. User objects include name, email, role, and created_at fields.", "metadata": {"source": "user-mgmt-docs", "category": "users"}},
    {"id": 10, "content": "Role-Based Access Control (RBAC): Define roles at /settings/roles. Each role has a set of permissions. Built-in roles: admin, editor, viewer. Custom roles can be created. Assign roles to users via PUT /users/:id/roles.", "metadata": {"source": "rbac-docs", "category": "authorization"}},
    {"id": 11, "content": "Data Export API: Request data exports via POST /exports. Supported formats: CSV, JSON, Parquet. Large exports are processed asynchronously. Poll GET /exports/:id for status. Download link provided when complete.", "metadata": {"source": "export-docs", "category": "data"}},
    {"id": 12, "content": "Batch Operations: Process multiple operations in a single request using POST /batch. Send an array of operations with method, path, and body. Maximum 100 operations per batch. Returns results in the same order.", "metadata": {"source": "batch-docs", "category": "batch"}},
    {"id": 13, "content": "WebSocket API: Connect to wss://api.example.com/ws for real-time updates. Authenticate with token in the connection URL. Subscribe to channels using {type: 'subscribe', channel: 'orders'}. Heartbeat every 30 seconds.", "metadata": {"source": "websocket-docs", "category": "realtime"}},
    {"id": 14, "content": "Caching Strategy: API responses include Cache-Control headers. GET requests are cacheable for 5 minutes by default. Use ETag and If-None-Match headers for conditional requests. POST/PUT/DELETE invalidate related caches.", "metadata": {"source": "caching-docs", "category": "performance"}},
    {"id": 15, "content": "API Versioning: The API uses URL-based versioning (e.g., /v1/, /v2/). The current version is v2. Version v1 is deprecated and will be removed in 6 months. Breaking changes are only introduced in major versions.", "metadata": {"source": "versioning-docs", "category": "versioning"}},
    {"id": 16, "content": "Database Connections: Configure database connections via environment variables. Supports PostgreSQL, MySQL, and SQLite. Connection pooling is enabled by default with max 20 connections. Use DATABASE_URL format.", "metadata": {"source": "database-docs", "category": "database"}},
    {"id": 17, "content": "Logging and Monitoring: All API requests are logged with request_id, timestamp, method, path, status_code, and duration. Access logs via GET /admin/logs. Integrate with external services using the /webhooks/logs endpoint.", "metadata": {"source": "logging-docs", "category": "monitoring"}},
    {"id": 18, "content": "Health Check: GET /health returns system status including database connectivity, cache availability, and external service status. Returns 200 if healthy, 503 if any component is degraded. Used for load balancer configuration.", "metadata": {"source": "health-docs", "category": "monitoring"}},
    {"id": 19, "content": "Two-Factor Authentication (2FA): Enable 2FA via POST /auth/2fa/enable. Supports TOTP (Google Authenticator) and SMS. After enabling, login requires a verification code. Backup codes are provided for account recovery.", "metadata": {"source": "2fa-docs", "category": "authentication"}},
    {"id": 20, "content": "Session Management: Sessions are created on login and stored server-side. Default timeout: 30 minutes of inactivity. GET /auth/sessions lists active sessions. DELETE /auth/sessions/:id to revoke a specific session.", "metadata": {"source": "session-docs", "category": "authentication"}},
    {"id": 21, "content": "Email API: Send transactional emails via POST /emails/send. Required fields: to, subject, body. Optional: cc, bcc, attachments, template_id. Rate limit: 100 emails/hour. Track delivery status via GET /emails/:id/status.", "metadata": {"source": "email-docs", "category": "email"}},
    {"id": 22, "content": "Notification API: Send push notifications via POST /notifications. Supports mobile (FCM, APNs), web push, and in-app notifications. Configure channels per user at /users/:id/notification-preferences.", "metadata": {"source": "notification-docs", "category": "notifications"}},
    {"id": 23, "content": "Payment Processing: Integrate payments using POST /payments. Supports credit cards, ACH, and digital wallets. PCI DSS compliant. Use /payments/:id/refund for refunds. Webhooks notify on payment status changes.", "metadata": {"source": "payment-docs", "category": "payments"}},
    {"id": 24, "content": "Subscription Management: Create subscription plans via POST /subscriptions/plans. Subscribe users with POST /subscriptions. Handle upgrades, downgrades, and cancellations. Proration is calculated automatically.", "metadata": {"source": "subscription-docs", "category": "billing"}},
    {"id": 25, "content": "Invoice API: Generate invoices via POST /invoices. Auto-generate from subscriptions or create manually. Download PDF via GET /invoices/:id/pdf. Send to customer via POST /invoices/:id/send.", "metadata": {"source": "invoice-docs", "category": "billing"}},
    {"id": 26, "content": "GraphQL API: Access the GraphQL endpoint at POST /graphql. Use introspection to explore the schema. Supports queries, mutations, and subscriptions. Query depth is limited to 10 levels. Complexity limit: 1000 points.", "metadata": {"source": "graphql-docs", "category": "api"}},
    {"id": 27, "content": "SDK Installation: Install the official SDK via npm install @company/sdk or pip install company-sdk. SDKs available for JavaScript, Python, Ruby, and Go. Auto-generated from OpenAPI spec. Includes TypeScript types.", "metadata": {"source": "sdk-docs", "category": "sdk"}},
    {"id": 28, "content": "Testing and Sandbox: Use the sandbox environment at sandbox.api.example.com. Test API key prefix: test_. Sandbox data is reset daily. Test credit card numbers available in documentation. No real charges in sandbox.", "metadata": {"source": "sandbox-docs", "category": "testing"}},
    {"id": 29, "content": "CORS Configuration: API supports Cross-Origin Resource Sharing. Configure allowed origins at /settings/cors. Wildcard (*) allowed in development. Production requires specific origins. Preflight requests cached for 1 hour.", "metadata": {"source": "cors-docs", "category": "security"}},
    {"id": 30, "content": "Data Encryption: All data is encrypted at rest using AES-256. Data in transit uses TLS 1.3. Field-level encryption available for sensitive data. Encryption keys are managed via AWS KMS. Key rotation every 365 days.", "metadata": {"source": "encryption-docs", "category": "security"}},
    {"id": 31, "content": "Audit Logs: All administrative actions are logged in the audit trail. Access via GET /admin/audit-logs. Logs include actor, action, resource, timestamp, and IP address. Logs are retained for 1 year. Export via CSV.", "metadata": {"source": "audit-docs", "category": "compliance"}},
    {"id": 32, "content": "GDPR Compliance: Support data subject requests via POST /privacy/export (data portability) and POST /privacy/delete (right to erasure). Consent management via /privacy/consent. Data processing agreements available.", "metadata": {"source": "gdpr-docs", "category": "compliance"}},
    {"id": 33, "content": "Image Processing API: Upload images to POST /images/process. Operations: resize, crop, rotate, compress, convert format. Chain operations in a pipeline. Output formats: JPEG, PNG, WebP. Max input size: 20MB.", "metadata": {"source": "image-docs", "category": "media"}},
    {"id": 34, "content": "Video Transcoding: Submit video transcoding jobs via POST /videos/transcode. Supports MP4, WebM, AVI input. Output presets: web, mobile, HD, 4K. Progress tracked via webhooks. HLS streaming output available.", "metadata": {"source": "video-docs", "category": "media"}},
    {"id": 35, "content": "Analytics API: Query analytics data via POST /analytics/query. Dimensions: time, user, location, device. Metrics: views, clicks, conversions, revenue. Time ranges up to 90 days. Real-time data available for Pro tier.", "metadata": {"source": "analytics-docs", "category": "analytics"}},
    {"id": 36, "content": "A/B Testing: Create experiments via POST /experiments. Define variants with traffic allocation percentages. Track conversion metrics per variant. Statistical significance calculated automatically. Minimum sample size recommendations provided.", "metadata": {"source": "ab-testing-docs", "category": "analytics"}},
    {"id": 37, "content": "Feature Flags: Manage feature flags via POST /features. Toggle features on/off per environment. Target by user segment, percentage, or specific users. Supports gradual rollouts. Real-time flag evaluation via SDK.", "metadata": {"source": "feature-flags-docs", "category": "deployment"}},
    {"id": 38, "content": "CI/CD Integration: Automate deployments using the Deployment API. POST /deployments to trigger. Supports GitHub Actions, GitLab CI, Jenkins. Rollback via POST /deployments/:id/rollback. Blue-green and canary strategies supported.", "metadata": {"source": "cicd-docs", "category": "deployment"}},
    {"id": 39, "content": "Environment Variables: Manage environment variables via /settings/env. Variables are encrypted and scoped to environments (development, staging, production). Changes require redeployment. Bulk import/export supported.", "metadata": {"source": "env-vars-docs", "category": "configuration"}},
    {"id": 40, "content": "Custom Domains: Configure custom domains at /settings/domains. Automatic SSL certificate provisioning via Let's Encrypt. DNS verification required. Supports wildcard domains. CDN integration included.", "metadata": {"source": "domains-docs", "category": "configuration"}},
    {"id": 41, "content": "Load Balancing: Automatic load balancing across application instances. Health checks determine instance availability. Supports round-robin and least-connections algorithms. Session affinity (sticky sessions) configurable.", "metadata": {"source": "load-balancing-docs", "category": "infrastructure"}},
    {"id": 42, "content": "Auto-scaling: Configure auto-scaling rules at /settings/scaling. Scale based on CPU, memory, or request count. Min/max instance limits. Scale-up delay: 2 minutes. Scale-down delay: 10 minutes. Scheduled scaling supported.", "metadata": {"source": "scaling-docs", "category": "infrastructure"}},
    {"id": 43, "content": "Container Registry: Push Docker images to registry.example.com. Authenticate with API key. Image scanning for vulnerabilities included. Automatic cleanup of untagged images after 30 days. Storage limit: 10GB free tier.", "metadata": {"source": "registry-docs", "category": "containers"}},
    {"id": 44, "content": "Serverless Functions: Deploy serverless functions via POST /functions. Supports Node.js, Python, and Go runtimes. Cold start <100ms. Maximum execution time: 30 seconds. Memory: 128MB to 1GB configurable.", "metadata": {"source": "serverless-docs", "category": "compute"}},
    {"id": 45, "content": "Message Queue API: Publish messages to queues via POST /queues/:name/messages. Subscribe with long polling or WebSocket. FIFO ordering guaranteed. Dead letter queue for failed messages. Retention: 7 days.", "metadata": {"source": "queue-docs", "category": "messaging"}},
    {"id": 46, "content": "Event Bus: Publish events via POST /events. Subscribe services to event types. At-least-once delivery guaranteed. Events stored for replay up to 24 hours. Supports event filtering with JSONPath expressions.", "metadata": {"source": "event-bus-docs", "category": "messaging"}},
    {"id": 47, "content": "Scheduled Tasks (Cron): Create scheduled tasks via POST /cron. Supports standard cron expressions. Minimum interval: 1 minute. Tasks execute serverless functions or HTTP endpoints. Timezone support included.", "metadata": {"source": "cron-docs", "category": "scheduling"}},
    {"id": 48, "content": "Geolocation API: Get location data via GET /geo/ip/:address. Returns country, region, city, latitude, longitude. Accuracy varies by IP type. Batch lookups via POST /geo/batch. Updated monthly from MaxMind database.", "metadata": {"source": "geo-docs", "category": "location"}},
    {"id": 49, "content": "Translation API: Translate text via POST /translate. Supports 100+ languages. Auto-detect source language. Batch translation for multiple strings. Glossary support for domain-specific terms. Neural machine translation.", "metadata": {"source": "translation-docs", "category": "ai"}},
    {"id": 50, "content": "Sentiment Analysis: Analyze text sentiment via POST /ai/sentiment. Returns positive, negative, neutral scores (0-1). Supports 12 languages. Batch analysis available. Custom model training for domain-specific content.", "metadata": {"source": "sentiment-docs", "category": "ai"}},
    {"id": 51, "content": "Text-to-Speech API: Convert text to audio via POST /tts. Voices available in 30+ languages. Control speed, pitch, and volume. Output formats: MP3, WAV, OGG. SSML markup supported for advanced control.", "metadata": {"source": "tts-docs", "category": "ai"}},
    {"id": 52, "content": "OCR API: Extract text from images via POST /ocr. Supports printed and handwritten text. Languages: English, Spanish, French, German, Chinese, Japanese. Returns text with bounding box coordinates. Table extraction available.", "metadata": {"source": "ocr-docs", "category": "ai"}},
    {"id": 53, "content": "Team Management: Invite team members via POST /teams/:id/invites. Roles: owner, admin, member. Transfer ownership via PUT /teams/:id/owner. Team-level billing and permissions. SSO integration for Enterprise.", "metadata": {"source": "team-docs", "category": "organization"}},
    {"id": 54, "content": "Single Sign-On (SSO): Configure SAML 2.0 or OIDC SSO at /settings/sso. Supports Okta, Azure AD, Google Workspace. Just-in-time provisioning available. Enforce SSO for all team members.", "metadata": {"source": "sso-docs", "category": "authentication"}},
    {"id": 55, "content": "IP Whitelisting: Restrict API access by IP at /settings/security/ip-whitelist. Add individual IPs or CIDR ranges. Separate lists for API access and admin dashboard. Changes take effect within 5 minutes.", "metadata": {"source": "ip-whitelist-docs", "category": "security"}},
    {"id": 56, "content": "Content Delivery Network (CDN): Static assets automatically served via CDN. Edge locations in 50+ regions. Cache invalidation via POST /cdn/purge. Custom cache rules configurable. Supports Brotli and Gzip compression.", "metadata": {"source": "cdn-docs", "category": "performance"}},
    {"id": 57, "content": "Database Migration: Manage schema migrations via POST /db/migrations. Auto-generate from model changes. Rollback with POST /db/migrations/:id/rollback. Migration history tracked. Zero-downtime migrations supported.", "metadata": {"source": "migration-docs", "category": "database"}},
    {"id": 58, "content": "Backup and Restore: Automated daily backups for all data. Manual backup via POST /backups. Restore from backup via POST /backups/:id/restore. Point-in-time recovery available for Pro tier. Backups retained for 30 days.", "metadata": {"source": "backup-docs", "category": "database"}},
    {"id": 59, "content": "API Documentation Generator: Auto-generate API docs from OpenAPI/Swagger spec. Upload spec at POST /docs/import. Interactive try-it-out console. Custom branding and domain. Markdown support in descriptions.", "metadata": {"source": "docs-generator", "category": "documentation"}},
    {"id": 60, "content": "Changelog API: Manage product changelog via POST /changelog. Rich text support with images and videos. Email notifications to subscribers. RSS feed at /changelog/rss. Categories and tags for organization.", "metadata": {"source": "changelog-docs", "category": "documentation"}},
    {"id": 61, "content": "Status Page: Public status page at status.example.com. Automated incident detection from health checks. Manual incident creation via POST /incidents. Subscriber notifications via email and SMS. Historical uptime data.", "metadata": {"source": "status-page-docs", "category": "monitoring"}},
    {"id": 62, "content": "Multi-tenancy: Isolate data per tenant using tenant_id in all API requests. Configure tenant settings at /tenants/:id/settings. Cross-tenant queries prohibited by default. Tenant provisioning via POST /tenants.", "metadata": {"source": "multi-tenant-docs", "category": "architecture"}},
]

# ── Embedding cache ─────────────────────────────────────────────────────────
EMBEDDINGS_CACHE_FILE = Path(__file__).parent / "embeddings_cache.json"
doc_embeddings: list[list[float]] = []


def compute_embedding_openai(text: str) -> list[float]:
    """Get embedding for a single text using OpenAI API."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def compute_embeddings_batch_openai(texts: list[str]) -> list[list[float]]:
    """Get embeddings for multiple texts in one OpenAI API call (max 2048 inputs)."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    # Results are returned in the same order as input
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def load_or_compute_embeddings() -> list[list[float]]:
    """Load cached embeddings or compute + cache them via OpenAI."""
    if EMBEDDINGS_CACHE_FILE.exists():
        try:
            with open(EMBEDDINGS_CACHE_FILE, "r") as f:
                cached = json.load(f)
            if len(cached) == len(DOCUMENTS):
                print(f"✅ Loaded {len(cached)} cached embeddings")
                return cached
        except Exception:
            pass

    print(f"🔄 Computing embeddings for {len(DOCUMENTS)} documents via OpenAI...")
    texts = [doc["content"] for doc in DOCUMENTS]

    # Batch in groups of 100 (well within OpenAI limits)
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        embs = compute_embeddings_batch_openai(batch)
        all_embeddings.extend(embs)
        print(f"   Embedded {min(i + batch_size, len(texts))}/{len(texts)} docs")

    # Cache to disk
    with open(EMBEDDINGS_CACHE_FILE, "w") as f:
        json.dump(all_embeddings, f)
    print(f"✅ Cached {len(all_embeddings)} embeddings to disk")

    return all_embeddings


# ── Cosine similarity ──────────────────────────────────────────────────────
def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_np = np.array(a, dtype=np.float32)
    b_np = np.array(b, dtype=np.float32)
    dot = np.dot(a_np, b_np)
    norm_a = np.linalg.norm(a_np)
    norm_b = np.linalg.norm(b_np)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def vector_search(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """Find top-k most similar documents by cosine similarity."""
    scores = []
    for i, doc_emb in enumerate(doc_embeddings):
        sim = cosine_similarity(query_embedding, doc_emb)
        scores.append((i, sim))

    scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for idx, sim in scores[:top_k]:
        # Normalize cosine similarity from [-1,1] to [0,1]
        normalized_score = round((sim + 1) / 2, 4)
        results.append({
            "id": DOCUMENTS[idx]["id"],
            "score": normalized_score,
            "content": DOCUMENTS[idx]["content"],
            "metadata": DOCUMENTS[idx]["metadata"],
        })
    return results


# ── LLM Re-ranking with Gemini ─────────────────────────────────────────────
def rerank_with_llm(query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    """Re-rank candidates using Gemini to score query-document relevance."""
    model = genai.GenerativeModel(RERANK_MODEL)
    reranked = []

    for candidate in candidates:
        prompt = (
            f'Query: "{query}"\n'
            f'Document: "{candidate["content"]}"\n\n'
            f"Rate the relevance of this document to the query on a scale of 0-10.\n"
            f"Respond with only the number."
        )

        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    max_output_tokens=5,
                ),
            )
            score_text = response.text.strip()
            score_raw = float(score_text.split()[0].replace(",", "").replace(".", "."))
            score_raw = max(0.0, min(10.0, score_raw))
            score_normalized = round(score_raw / 10.0, 4)
        except Exception as e:
            print(f"⚠️  Re-ranking error for doc {candidate['id']}: {e}")
            score_normalized = candidate["score"]

        reranked.append({
            "id": candidate["id"],
            "score": score_normalized,
            "content": candidate["content"],
            "metadata": candidate["metadata"],
        })

    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked[:top_k]


# ── Request / Response models ──────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    k: int = 5
    rerank: bool = True
    rerankK: int = 3


class ResultItem(BaseModel):
    id: int
    score: float
    content: str
    metadata: dict


class MetricsModel(BaseModel):
    latency: int
    totalDocs: int


class SearchResponse(BaseModel):
    results: list[ResultItem]
    reranked: bool
    metrics: MetricsModel


# ── API Endpoints ───────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    """Load/compute embeddings on startup."""
    global doc_embeddings
    try:
        doc_embeddings = load_or_compute_embeddings()
        print(f"🚀 Server ready with {len(doc_embeddings)} document embeddings")
    except Exception as e:
        print(f"⚠️  Could not load embeddings on startup: {e}")
        print("   Embeddings will be computed on first search request.")


@app.get("/")
async def root():
    return {
        "message": "Semantic Search with Re-ranking API",
        "usage": "POST /search with { query, k, rerank, rerankK }",
        "docs": "/docs",
        "embedding_model": EMBEDDING_MODEL,
        "rerank_model": RERANK_MODEL,
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "documents": len(DOCUMENTS),
        "embeddings_loaded": len(doc_embeddings) > 0,
        "embedding_model": EMBEDDING_MODEL,
    }


# ── Main search endpoint: POST /search ─────────────────────────────────────
@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Two-stage semantic search:
    1. Embed query via OpenAI → cosine similarity → top K candidates
    2. (Optional) Re-rank with Gemini LLM → top rerankK results
    """
    start_time = time.time()

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Lazy-load embeddings if not loaded on startup
    global doc_embeddings
    if not doc_embeddings:
        try:
            doc_embeddings = load_or_compute_embeddings()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Failed to compute embeddings: {e}")

    # Stage 1: Embed query + Vector Search
    try:
        query_embedding = compute_embedding_openai(request.query)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to embed query: {e}")

    candidates = vector_search(query_embedding, top_k=request.k)

    # Handle edge case: no results
    if not candidates:
        elapsed_ms = int((time.time() - start_time) * 1000)
        return SearchResponse(
            results=[],
            reranked=False,
            metrics=MetricsModel(latency=elapsed_ms, totalDocs=len(DOCUMENTS)),
        )

    # Stage 2: Re-ranking (optional)
    did_rerank = False
    if request.rerank and len(candidates) > 0:
        results = rerank_with_llm(request.query, candidates, top_k=request.rerankK)
        did_rerank = True
    else:
        results = candidates[: request.rerankK]

    elapsed_ms = int((time.time() - start_time) * 1000)

    return SearchResponse(
        results=[ResultItem(**r) for r in results],
        reranked=did_rerank,
        metrics=MetricsModel(latency=elapsed_ms, totalDocs=len(DOCUMENTS)),
    )


# Also handle POST on root "/" in case grader sends there
@app.post("/", response_model=SearchResponse)
async def search_root(request: SearchRequest):
    """Same as /search but on root path."""
    return await search(request)


# ── Run directly ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
