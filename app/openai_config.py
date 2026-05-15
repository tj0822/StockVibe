"""
AI 모델 설정 및 그룹 정의
- Claude 모델 (Anthropic) — ANTHROPIC_API_KEY 필요
- 무료/저가 OpenAI 모델 (Freemium)
- 유료 프리미엄 OpenAI 모델 (Premium)
- 커스텀/신규 모델 (Custom)

.env 파일에서 OPENAI_CUSTOM_MODELS에 새 모델을 추가하면 자동으로 인식됩니다.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ===== Claude 모델 (Anthropic) =====
CLAUDE_MODELS = {
    "claude-opus-4-7": "Claude Opus 4.7 (최고 성능, 복잡한 분석)",
    "claude-sonnet-4-6": "Claude Sonnet 4.6 (균형잡힌 성능, 추천)",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5 (빠르고 경제적)",
}

# ===== 무료/저가 모델 (기본) =====
FREEMIUM_MODELS = {
    "gpt-3.5-turbo": "GPT-3.5 Turbo (빠르고 저렴)",
    "gpt-4o-mini": "GPT-4o Mini (균형잡힌 선택)",
}

# ===== 유료 프리미엄 모델 (고급) =====
PREMIUM_MODELS = {
    "gpt-4": "GPT-4 (기본 고성능)",
    "gpt-4-turbo": "GPT-4 Turbo (고속 고성능)",
    "gpt-4o": "GPT-4o (최신 멀티모달)",
    "o1": "o1 (추론 최적화, 예약 필요)",
}

# ===== 커스텀 모델 (사용자 정의 - 신규/실험 모델) =====
CUSTOM_MODELS = {}

# 전체 모델 맵
ALL_MODELS = {}


def is_claude_model(model_name: str) -> bool:
    """모델이 Claude(Anthropic) 모델인지 확인"""
    return model_name.startswith("claude-")


def _parse_models_from_env(env_key: str) -> list:
    """환경변수에서 모델 리스트 파싱 (쉼표로 구분)"""
    env_value = os.getenv(env_key, "")
    if not env_value:
        return []
    return [m.strip() for m in env_value.split(",") if m.strip()]


def _init_models_from_env():
    """환경변수로부터 모델 그룹 초기화"""
    global FREEMIUM_MODELS, PREMIUM_MODELS, CUSTOM_MODELS, ALL_MODELS
    
    # 무료 모델 오버라이드
    _env_freemium = _parse_models_from_env("OPENAI_FREEMIUM_MODELS")
    if _env_freemium:
        FREEMIUM_MODELS = {}
        for model_name in _env_freemium:
            # 모델 설명 생성 (AI 모델명)
            FREEMIUM_MODELS[model_name] = f"{model_name} (무료)"
    
    # 유료 모델 오버라이드
    _env_premium = _parse_models_from_env("OPENAI_PREMIUM_MODELS")
    if _env_premium:
        PREMIUM_MODELS = {}
        for model_name in _env_premium:
            PREMIUM_MODELS[model_name] = f"{model_name} (유료)"
    
    # 커스텀 모델 추가 (신규/실험 모델)
    _env_custom = _parse_models_from_env("OPENAI_CUSTOM_MODELS")
    if _env_custom:
        CUSTOM_MODELS = {}
        for model_name in _env_custom:
            CUSTOM_MODELS[model_name] = f"{model_name} ⭐ (신규/실험)"
    
    # 최종 전체 모델 맵 생성 (Claude 먼저)
    ALL_MODELS.clear()
    ALL_MODELS.update(CLAUDE_MODELS)
    ALL_MODELS.update(FREEMIUM_MODELS)
    ALL_MODELS.update(PREMIUM_MODELS)
    ALL_MODELS.update(CUSTOM_MODELS)


# 초기화 실행
_init_models_from_env()


def get_default_model() -> str:
    """기본 모델 반환 (환경변수 또는 하드코딩)"""
    return os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")


def get_model_options() -> dict:
    """{모델_이름: 설명} 형식의 전체 모델 옵션 반환"""
    return ALL_MODELS.copy()


def get_claude_options() -> dict:
    """Claude(Anthropic) 모델 옵션 반환"""
    return CLAUDE_MODELS.copy()


def get_freemium_options() -> dict:
    """무료/저가 모델 옵션 반환"""
    return FREEMIUM_MODELS.copy()


def get_premium_options() -> dict:
    """유료 프리미엄 모델 옵션 반환"""
    return PREMIUM_MODELS.copy()


def get_custom_options() -> dict:
    """커스텀 모델 옵션 반환"""
    return CUSTOM_MODELS.copy()


def get_model_label(model_name: str) -> str:
    """모델 이름에 대한 레이블 반환 (없으면 모델명 자체)"""
    return ALL_MODELS.get(model_name, model_name)


def is_premium_model(model_name: str) -> bool:
    """모델이 프리미엄 모델인지 확인"""
    return model_name in PREMIUM_MODELS


def is_freemium_model(model_name: str) -> bool:
    """모델이 무료/저가 모델인지 확인"""
    return model_name in FREEMIUM_MODELS


def is_custom_model(model_name: str) -> bool:
    """모델이 커스텀 모델인지 확인"""
    return model_name in CUSTOM_MODELS


def add_custom_model(model_name: str, description: str = None) -> None:
    """런타임에 커스텀 모델 추가"""
    global ALL_MODELS, CUSTOM_MODELS
    
    if not description:
        description = f"{model_name} ⭐ (사용자 추가)"
    
    CUSTOM_MODELS[model_name] = description
    ALL_MODELS[model_name] = description


def list_all_models() -> list:
    """모든 사용 가능한 모델 리스트 반환"""
    return sorted(list(ALL_MODELS.keys()))
