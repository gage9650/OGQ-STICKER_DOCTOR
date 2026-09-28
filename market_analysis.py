"""OGQ 시장 데이터와 사용자 입력을 실제 이미지와 함께 비교 분석한다."""
from __future__ import annotations

import json
from typing import Any

import requests
from PIL import Image
import io


MARKET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "similarity_level": {
            "type": "string",
            "enum": ["낮음", "보통", "높음", "매우 높음"],
            "description": "사용자 스티커와 검색된 OGQ 콘텐츠 사이의 실제 시장 유사성 정도. 비슷한 느낌뿐 아니라 실제 이미지/콘셉트/태그를 종합하되, 시각적 복제 여부를 단정하지 않는다.",
        },
        "similarity_summary": {
            "type": "string",
            "description": "시장 유사성을 1~2문장으로 구체적으로 설명한다.",
        },
        "comparisons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "result_index": {"type": "integer"},
                    "similarity_level": {
                        "type": "string",
                        "enum": ["낮음", "보통", "높음", "매우 높음"],
                    },
                    "similarity_reason": {"type": "string"},
                    "same_points": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "different_points": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "result_index",
                    "similarity_level",
                    "similarity_reason",
                    "same_points",
                    "different_points",
                ],
            },
            "description": "검색 결과별 구체적인 비교. 실제 확인 가능한 결과만 최대 4개.",
        },
        "similarities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "여러 시장 결과에서 반복적으로 확인되는 공통 요소.",
        },
        "differences": {
            "type": "array",
            "items": {"type": "string"},
            "description": "사용자 스티커와 시장 결과가 실제로 구별되는 요소. 사용자 입력 태그만으로 차이를 만들어내지 않는다.",
        },
        "strengths": {
            "type": "array",
            "items": {"type": "string"},
            "description": "시장 결과와 비교했을 때 사용자의 콘셉트에서 확인되는 강점.",
        },
        "gaps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "시장 결과에서 반복적으로 확인되는 요소와 비교해 보완할 수 있는 부분.",
        },
        "priority_actions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "제작자가 다음 수정에서 먼저 확인할 행동을 최대 3개.",
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
            "description": "분석의 근거가 된 OGQ 검색 결과 번호와 구체적인 이유.",
        },
    },
    "required": [
        "similarity_level",
        "similarity_summary",
        "comparisons",
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
                "main_image_url": item.get("main_image_url", ""),
                "visual_match_hint": item.get("visual_match_hint", ""),
            }
        )
    return compact


def build_market_context(user_feelings: str, user_tags: list[str], results: list[dict]) -> str:
    return json.dumps(
        {
            "user_feelings": user_feelings,
            "user_tags": user_tags,
            "ogq_market_results": _compact_results(results),
        },
        ensure_ascii=False,
        indent=2,
    )


def build_market_analysis_prompt(user_feelings: str, user_tags: list[str], results: list[dict]) -> str:
    context = build_market_context(user_feelings, user_tags, results)
    return f"""당신은 스티커 제작자의 시장조사를 도와주는 분석 AI입니다.
아래 '사용자 스티커 이미지'와 OGQ 마켓 검색 결과를 함께 보고 비교하세요.

[사용자 설명]
느낌/분위기: {user_feelings or '입력 없음'}
태그: {', '.join(user_tags) if user_tags else '입력 없음'}

[OGQ 마켓 검색 결과]
{context}

핵심 원칙:
1. 이것은 시장 참고 분석이지 OGQ 내부 심사 결과나 합격 확률 예측이 아닙니다.
2. 사용자 태그는 사용자가 제시한 '가설/메타데이터'일 뿐입니다. 사용자가 입력한 태그만으로 이미지의 실제 소재나 차이점을 단정하지 마세요.
3. 사용자 이미지와 시장 참조 이미지가 실제로 비슷하면 그 사실을 명확하게 인정하세요. 특히 동일하거나 거의 동일해 보이는 참조가 있으면 '매우 높음'으로 평가할 수 있습니다.
4. 시장 결과와의 차이를 말할 때는 반드시 실제 이미지, 제목, 설명, 태그 중 하나 이상의 근거를 제시하세요.
5. '검색 결과는 곰이고 사용자는 강아지다'처럼 태그만 보고 차이를 만들어내지 마세요. 실제 사용자 이미지가 강아지를 보여주는지 먼저 확인하세요.
6. 각 비교는 result_index를 정확히 연결해야 합니다.
7. similarities/differences/strengths/gaps는 검색 결과의 반복 패턴을 중심으로 작성하고, 확인되지 않은 시각적 특징은 상상하지 마세요.
8. evidence는 실제 근거가 있는 결과만 사용하세요.
9. priority_actions는 최대 3개입니다.
"""


def _normalise_text_list(value: Any, limit: int) -> list[str]:
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


def normalize_market_analysis(data: dict[str, Any] | None) -> dict[str, Any]:
    data = data if isinstance(data, dict) else {}
    level = str(data.get("similarity_level") or "보통").strip()
    if level not in {"낮음", "보통", "높음", "매우 높음"}:
        level = "보통"

    comparisons: list[dict[str, Any]] = []
    raw = data.get("comparisons")
    if isinstance(raw, list):
        for item in raw[:4]:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("result_index"))
            except (TypeError, ValueError):
                continue
            if not 1 <= idx <= 8:
                continue
            cmp_level = str(item.get("similarity_level") or "보통").strip()
            if cmp_level not in {"낮음", "보통", "높음", "매우 높음"}:
                cmp_level = "보통"
            comparisons.append(
                {
                    "result_index": idx,
                    "similarity_level": cmp_level,
                    "similarity_reason": " ".join(str(item.get("similarity_reason") or "").split()),
                    "same_points": _normalise_text_list(item.get("same_points"), 4),
                    "different_points": _normalise_text_list(item.get("different_points"), 4),
                }
            )

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
        "comparisons": comparisons,
        "similarities": _normalise_text_list(data.get("similarities"), 6),
        "differences": _normalise_text_list(data.get("differences"), 6),
        "strengths": _normalise_text_list(data.get("strengths"), 5),
        "gaps": _normalise_text_list(data.get("gaps"), 6),
        "priority_actions": _normalise_text_list(data.get("priority_actions"), 3),
        "evidence": evidence,
    }


def build_market_context_for_diagnosis(market_analysis: dict[str, Any], results: list[dict]) -> str:
    payload = {
        "market_analysis": {
            "similarity_level": market_analysis.get("similarity_level", ""),
            "similarity_summary": market_analysis.get("similarity_summary", ""),
            "comparisons": list(market_analysis.get("comparisons") or [])[:4],
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


def _visual_hash(image_bytes: bytes) -> list[int] | None:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        bg = Image.new("RGBA", image.size, "white")
        bg.alpha_composite(image)
        gray = bg.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
        pixels = list(gray.getdata())
        mean = sum(pixels) / len(pixels)
        return [1 if px >= mean else 0 for px in pixels]
    except Exception:
        return None


def _hash_similarity(a: list[int] | None, b: list[int] | None) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    same = sum(x == y for x, y in zip(a, b))
    return same / len(a)


def annotate_visual_match_hints(user_image_bytes: bytes, results: list[dict], threshold: float = 0.93) -> list[dict]:
    """다운로드 가능한 OGQ 썸네일과 사용자 이미지 사이의 강한 시각 일치 신호를 표시한다.

    최종 판단은 AI가 하며, 이 값은 AI에게 전달하는 보조 신호일 뿐이다.
    """
    user_hash = _visual_hash(user_image_bytes)
    for item in results:
        item["visual_match_hint"] = ""
        url = item.get("main_image_url") or item.get("thumbnail_url")
        if not url or not user_hash:
            continue
        try:
            r = requests.get(url, timeout=6)
            r.raise_for_status()
            score = _hash_similarity(user_hash, _visual_hash(r.content))
            if score >= threshold:
                item["visual_match_hint"] = f"강한 시각적 일치 신호({score:.0%})"
            elif score >= 0.85:
                item["visual_match_hint"] = f"유사한 시각적 패턴 신호({score:.0%})"
        except Exception:
            continue
    return results


def download_market_reference_images(
    results: list[dict],
    *,
    max_results: int = 4,
    timeout: int = 6,
    max_bytes: int = 1_500_000,
) -> list[dict[str, Any]]:
    """Gemini 시장 비교용으로 소수의 참조 이미지만 다운로드한다.

    API의 다운로드 토큰 엔드포인트는 사용하지 않고, 검색 결과에 이미 제공되는
    thumbnail/image URL만 사용한다. 실패한 이미지는 건너뛰고 텍스트 메타데이터 비교는 유지한다.
    """
    references: list[dict[str, Any]] = []
    for index, item in enumerate(results[:max_results], start=1):
        url = ""
        images = item.get("images")
        if isinstance(images, list) and images:
            first = images[0]
            if isinstance(first, dict):
                url = str(first.get("imageUrl") or first.get("thumbnailUrl") or "").strip()
        if not url:
            url = str(item.get("main_image_url") or item.get("thumbnail_url") or "").strip()
        if not url:
            continue
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            if len(response.content) > max_bytes:
                continue
            mime = response.headers.get("content-type", "image/png").split(";", 1)[0].strip()
            if not mime.startswith("image/"):
                mime = "image/png"
            references.append(
                {
                    "index": index,
                    "title": str(item.get("title") or "OGQ 콘텐츠"),
                    "bytes": response.content,
                    "mime_type": mime,
                }
            )
        except Exception:
            continue
    return references
