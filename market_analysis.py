"""OGQ 시장 데이터와 사용자 입력을 구조화해서 비교 분석한다."""
from __future__ import annotations

import json
from typing import Any


MARKET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "similarity_level": {
            "type": "string",
            "enum": ["낮음", "보통", "높음"],
            "description": "사용자 입력과 OGQ 검색 결과 사이에서 확인되는 시장 유사성의 정도. 시각적 복제 여부를 의미하지 않음.",
        },
        "similarity_summary": {
            "type": "string",
            "description": "시장 유사성을 1~2문장으로 설명한다.",
        },
        "similarities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "사용자 입력과 OGQ 검색 결과에서 실제로 공통으로 확인되는 요소들.",
        },
        "differences": {
            "type": "array",
            "items": {"type": "string"},
            "description": "검색 결과와 구별되는 요소. 검색 데이터로 뒷받침되는 범위만 작성.",
        },
        "strengths": {
            "type": "array",
            "items": {"type": "string"},
            "description": "시장 결과와 비교했을 때 사용자의 콘셉트가 가질 수 있는 구체적인 장점.",
        },
        "gaps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "시장 결과에서 반복적으로 확인되는 요소와 비교했을 때 보완하면 좋은 부분.",
        },
        "priority_actions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "제작자가 다음 수정에서 먼저 확인할 3가지.",
        },
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "result_index": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["result_index", "reason"],
            },
            "description": "분석의 근거가 된 OGQ 검색 결과 번호와 이유.",
        },
    },
    "required": [
        "similarity_level",
        "similarity_summary",
        "similarities",
        "differences",
        "strengths",
        "gaps",
        "priority_actions",
        "evidence",
    ],
}


def _compact_results(results: list[dict]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for i, item in enumerate(results[:8], start=1):
        compact.append(
            {
                "index": i,
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "creator": item.get("creator_name", ""),
                "matched_keyword": item.get("matched_keyword", ""),
                "tags": list(item.get("tags") or [])[:10],
                "published_at": item.get("published_at", ""),
            }
        )
    return compact


def build_market_context(
    user_feelings: str,
    user_tags: list[str],
    results: list[dict],
) -> str:
    """시장 검색 결과를 짧고 안정적인 JSON 컨텍스트로 만든다."""
    return json.dumps(
        {
            "user_feelings": user_feelings,
            "user_tags": user_tags,
            "ogq_market_results": _compact_results(results),
        },
        ensure_ascii=False,
        indent=2,
    )


def build_market_analysis_prompt(
    user_feelings: str,
    user_tags: list[str],
    results: list[dict],
) -> str:
    context = build_market_context(user_feelings, user_tags, results)
    return f"""당신은 스티커 제작자의 시장조사를 도와주는 분석 AI입니다.
아래 사용자 설명과 OGQ 마켓 검색 결과만 근거로 분석하세요.

[사용자 설명]
느낌/분위기: {user_feelings or '입력 없음'}
태그: {', '.join(user_tags) if user_tags else '입력 없음'}

[OGQ 마켓 검색 결과]
{context}

중요한 원칙:
1. 이것은 '시장 유사성 참고 분석'이지 OGQ 내부 심사 결과나 합격 확률 예측이 아닙니다.
2. 실제 검색 결과의 제목·설명·태그·검색어만 근거로 판단하세요.
3. 시장 결과에 없는 시각적 특징은 추측해서 쓰지 마세요.
4. '비슷하다'는 표현은 콘셉트·상황·태그·표현 주제의 유사성을 중심으로 사용하세요.
5. 시장에서 뒤떨어진다고 단정하지 말고, '보완 여지가 있다'는 식으로 구체적으로 설명하세요.
6. differences와 strengths는 사용자 입력과 검색 데이터가 함께 뒷받침하는 범위에서만 작성하세요.
7. evidence의 result_index는 위 OGQ 검색 결과 번호(1~8) 중 실제 근거가 있는 번호만 사용하세요.
8. priority_actions는 정확히 3개 이내로 작성하세요.
"""


def build_market_context_for_diagnosis(
    market_analysis: dict[str, Any],
    results: list[dict],
) -> str:
    """이미지 진단 AI에 넘길 '시장 분석 결과'를 만든다.

    기존처럼 분석 프롬프트를 재사용하지 않고, 실제 생성된 분석 결과와
    그 분석의 근거 콘텐츠만 전달한다.
    """
    payload = {
        "market_analysis": {
            "similarity_level": market_analysis.get("similarity_level", ""),
            "similarity_summary": market_analysis.get("similarity_summary", ""),
            "similarities": list(market_analysis.get("similarities") or [])[:6],
            "differences": list(market_analysis.get("differences") or [])[:6],
            "strengths": list(market_analysis.get("strengths") or [])[:5],
            "gaps": list(market_analysis.get("gaps") or [])[:6],
            "priority_actions": list(market_analysis.get("priority_actions") or [])[:3],
            "evidence": list(market_analysis.get("evidence") or [])[:8],
        },
        "ogq_market_results": _compact_results(results),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def normalize_market_analysis(data: dict[str, Any] | None) -> dict[str, Any]:
    """Gemini 결과를 UI에서 안전하게 사용할 수 있는 최소 구조로 정규화."""
    data = data if isinstance(data, dict) else {}

    level = str(data.get("similarity_level") or "보통").strip()
    if level not in {"낮음", "보통", "높음"}:
        level = "보통"

    def clean_list(value: Any, limit: int) -> list[str]:
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            text = " ".join(str(item or "").split())
            if text and text not in out:
                out.append(text)
            if len(out) >= limit:
                break
        return out

    evidence: list[dict[str, Any]] = []
    raw_evidence = data.get("evidence")
    if isinstance(raw_evidence, list):
        for item in raw_evidence[:8]:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("result_index"))
            except (TypeError, ValueError):
                continue
            if not 1 <= idx <= 8:
                continue
            reason = " ".join(str(item.get("reason") or "").split())
            if reason:
                evidence.append({"result_index": idx, "reason": reason})

    return {
        "similarity_level": level,
        "similarity_summary": " ".join(str(data.get("similarity_summary") or "").split()),
        "similarities": clean_list(data.get("similarities"), 6),
        "differences": clean_list(data.get("differences"), 6),
        "strengths": clean_list(data.get("strengths"), 5),
        "gaps": clean_list(data.get("gaps"), 6),
        "priority_actions": clean_list(data.get("priority_actions"), 3),
        "evidence": evidence,
    }
