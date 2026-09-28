"""사용자 입력 + OGQ 검색 결과를 AI가 비교하기 위한 정리/프롬프트."""
from __future__ import annotations

import json


def build_market_context(user_feelings: str, user_tags: list[str], results: list[dict]) -> str:
    """OGQ 결과를 너무 길지 않게 AI 컨텍스트로 만든다."""
    compact = []
    for i, item in enumerate(results[:20], start=1):
        compact.append(
            {
                "index": i,
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "creator": item.get("creator_name", ""),
                "matched_keyword": item.get("matched_keyword", ""),
            }
        )
    return json.dumps(
        {
            "user_feelings": user_feelings,
            "user_tags": user_tags,
            "ogq_market_results": compact,
        },
        ensure_ascii=False,
        indent=2,
    )


def build_market_analysis_prompt(user_feelings: str, user_tags: list[str], results: list[dict]) -> str:
    context = build_market_context(user_feelings, user_tags, results)
    return f"""아래는 사용자가 설명한 자신의 스티커 특성과 OGQ 마켓 검색 결과입니다.

[사용자 설명]
느낌/분위기: {user_feelings or '입력 없음'}
태그: {', '.join(user_tags) if user_tags else '입력 없음'}

[OGQ 마켓 검색 결과]
{context}

이 자료를 근거로 한국어 존댓말로 분석하세요.

반드시 다음 구조를 사용하세요.

### 시장에서 비슷하게 보이는 이유
사용자 입력과 OGQ 검색 결과에서 공통으로 보이는 느낌·상황·표현을 설명하세요.

### 기존 콘텐츠와 다른 점
사용자 스티커가 기존 결과와 구별되는 요소를 설명하세요. 데이터에 없는 차별점은 추측하지 마세요.

### 현재 스티커의 장점
시장 검색 결과와 비교했을 때 사용자 스티커가 가질 수 있는 구체적인 강점을 설명하세요.

### 보완하면 좋은 점
시장 결과에서 반복적으로 보이는 요소와 비교해 부족해 보일 수 있는 부분을 설명하세요.
단, '심사에서 반드시 탈락한다'거나 '판매가 안 된다'고 단정하지 마세요.

### 가장 먼저 확인할 것
제작자가 다음 수정에서 우선적으로 확인할 3가지를 순서대로 제시하세요.

주의:
- 이것은 시장 참고 분석이지 OGQ 내부 심사 결과 예측이 아닙니다.
- OGQ 검색 API에서 받은 제목/설명 등 확인 가능한 데이터만 근거로 사용하세요.
"""
