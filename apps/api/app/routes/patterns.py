from fastapi import APIRouter
from app.settings import settings
from app.services.pattern_registry import load_patterns

router = APIRouter(prefix="/patterns", tags=["patterns"])

@router.get("")
def get_patterns():
    patterns = load_patterns(settings.PATTERN_DIR)
    return {
        "patterns": [
            {"id": p.id, "name": p.name}
            for p in patterns
        ]
    }
