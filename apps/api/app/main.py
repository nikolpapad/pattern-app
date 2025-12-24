from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.settings import settings
from app.routes.patterns import router as patterns_router
from app.routes.extract import router as extract_router

app = FastAPI(title=settings.APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.ALLOWED_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "ok": True,
        "message": "Pattern Extractor API is running",
        "routes": ["/health", "/patterns", "/extract/pdf", "/docs"]
    }

@app.get("/health")
def health():
    return {"ok": True}

app.include_router(patterns_router)
app.include_router(extract_router)

