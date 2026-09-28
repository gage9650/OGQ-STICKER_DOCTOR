# diagnosis.py — Gemini AI 스티커 진단
from __future__ import annotations

import json
import re
from typing import Any

from google import genai
from google.genai import types


DIAGNOSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "한 문장짜리 총평. 80~120자 정도.",
        },
        "detail": {
            "type": "string",
            "description": "총평을 뒷받침하는 구체적인 상세 설명. 3~5문장.",
        },
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                    "area": {"type": "string"},
                    "what": {"type": "string"},
                    "why": {"type": "string"},
                    "how": {"type": "string"},
                    "market_basis": {"type": "string"},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                },
                "required": [
                    "severity",
                    "area",
                    "what",
                    "why",
                    "how",
                    "market_basis",
                    "bbox",
                ],
            },
            "description": "실제로 개선할 가치가 있는 문제만 최대 4개.",
        },
        "strengths": {
            "type": "array",
            "items": {"type": "string"},
            "description": "이미지에서 실제로 확인되는 구체적인 장점 2~4개.",
        },
        "market_note": {
            "type": "string",
            "description": "시장 분석과 연결되는 핵심 참고사항. 없으면 빈 문자열.",
        },
    },
    "required": ["summary", "detail", "findings", "strengths", "market_note"],
}


SYSTEM_PROMPT = """당신은 스티커 콘텐츠를 검토하는 AI 진단 도구입니다.
공식적으로 공개된 OGQ 제작 가이드와 사용자가 선택한 검사 기준을 바탕으로 분석하세요.

중요:
- OGQ의 비공개 내부 심사 매뉴얼을 알고 있다고 주장하지 마세요.
- 심사 통과 확률을 예측하지 마세요.
- 이미지에서 실제로 확인할 수 없는 사실은 지어내지 마세요.
- 문제는 가능한 경우 이미지 안의 실제 위치를 bbox로 표시하세요.
- bbox 좌표는 0~1000 정규화 좌표이며 [left, top, right, bottom] 순서입니다.
- 문제 위치를 특정하기 어렵다면 bbox는 []로 두세요.
- 전체 이미지를 bbox로 지정하는 것은 해당 문제가 정말 이미지 전체에 걸쳐 있을 때만 허용합니다.
- 각 finding의 why에는 가능하면 시장 분석 자료에서 확인된 근거를 연결하세요.
- 시장 자료가 없거나 근거가 약하면 일반적인 디자인 원칙과 공개 가이드 기준임을 분명히 하세요.
- 한 줄 총평은 반드시 한 문장만 쓰세요.
- detail은 summary를 반복하지 말고, 이미지에서 실제로 확인되는 근거를 중심으로 3~5문장으로 설명하세요.
"""


def _extract_json(text: str) -> dict[str, Any]:
    """JSON 응답을 최대한 안전하게 추출한다."""
    text = (text or "").strip()
    if not text:
        return {}

    candidates = [text]

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        candidates.append(text[first : last + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    return {}


def _normalise_bbox(bbox: Any) -> list[int]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return []
    try:
        values = [max(0, min(1000, int(float(v)))) for v in bbox]
        if values[2] <= values[0] or values[3] <= values[1]:
            return []
        # 지나치게 큰 bbox는 전체 이미지 표시 오류일 가능성이 높아 보수적으로 버린다.
        area_ratio = ((values[2] - values[0]) * (values[3] - values[1])) / 1_000_000
        if area_ratio > 0.97:
            return []
        return values
    except (TypeError, ValueError):
        return []


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())


def _one_line_summary(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return "이미지의 주요 특징과 선택한 검사 기준을 종합해 개선 우선순위를 확인할 필요가 있습니다."
    # 첫 문장만 사용해 '한 줄 총평' 역할을 고정한다.
    parts = re.split(r"(?<=[.!?다요함])\s+", text)
    summary = parts[0].strip()
    if len(summary) > 150:
        summary = summary[:147].rstrip(" ,.;") + "…"
    return summary


def diagnose_detailed(
    file_bytes: bytes,
    media_type: str,
    api_key: str,
    selected_criteria: list[str] | None = None,
    market_context: str = "",
) -> dict[str, Any]:
    """한 장의 스티커를 구조화된 진단 결과로 분석한다."""
    client = genai.Client(api_key=api_key)
    criteria = selected_criteria or [
        "가독성", "다크모드", "오탈자", "콘텐츠 적합성", "여백", "중복·유사성"
    ]

    user_prompt = f"""
이 이미지를 아래 선택된 검사 기준으로 진단하세요.

[선택된 검사 기준]
{json.dumps(criteria, ensure_ascii=False)}

[시장 비교 결과]
{market_context or "시장 비교를 실행하지 않았습니다."}

다음 JSON 객체 하나만 반환하세요.

- summary: 정확히 1문장의 한 줄 총평
- detail: summary를 반복하지 않는 3~5문장의 상세 설명
- findings: 실제로 개선 가치가 있는 항목만 최대 4개
- strengths: 이미지에서 실제로 확인되는 구체적인 장점 2~4개
- market_note: 시장 비교가 있을 때 핵심만 요약

각 finding은 다음 구조입니다.
- severity: high|medium|low
- area: 문제 영역
- what: 실제로 어디에서 무엇이 문제인지
- why: 왜 확인해야 하는지. 시장 분석에 근거가 있으면 구체적으로 연결
- how: 제작자가 바로 실행할 수 있는 수정 기획안
- market_basis: 시장 분석과 연결되는 경우에만 짧게 근거를 적고, 없으면 빈 문자열
- bbox: [left, top, right, bottom] 정규화 좌표. 특정하기 어렵다면 []

절대로 모든 문제에 bbox=[0,0,1000,1000]을 사용하지 마세요.
"""

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=[
            types.Part.from_bytes(data=file_bytes, mime_type=media_type),
            user_prompt,
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=DIAGNOSIS_SCHEMA,
            max_output_tokens=2200,
        ),
    )

    raw_text = response.text or ""
    data = _extract_json(raw_text)
    if not data:
        return {
            "summary": "AI 진단 결과를 구조화하지 못했습니다.",
            "detail": "AI가 응답을 반환했지만 결과 형식이 예상과 달라 상세 항목으로 정리하지 못했습니다. 다시 진단하면 더 안정적인 결과를 받을 수 있습니다.",
            "findings": [],
            "strengths": [],
            "market_note": "",
            "raw_text": raw_text,
        }

    findings: list[dict[str, Any]] = []
    for item in (data.get("findings") or [])[:4]:
        if not isinstance(item, dict):
            continue
        findings.append(
            {
                "severity": item.get("severity", "low"),
                "area": _clean_text(item.get("area") or "검토 항목"),
                "what": _clean_text(item.get("what")),
                "why": _clean_text(item.get("why")),
                "how": _clean_text(item.get("how")),
                "market_basis": _clean_text(item.get("market_basis")),
                "bbox": _normalise_bbox(item.get("bbox")),
            }
        )

    strengths = []
    for value in (data.get("strengths") or [])[:4]:
        cleaned = _clean_text(value)
        if cleaned and cleaned not in strengths:
            strengths.append(cleaned)

    detail = str(data.get("detail") or "").strip()
    detail = re.sub(r"\s+", " ", detail)
    if len(detail) < 30:
        detail = "이미지에서 확인되는 시각적 특성과 선택한 검사 기준을 함께 고려해 판단했습니다. 문제가 있는 항목은 실제 수정 단계에서 우선순위를 정해 반영하는 것이 좋습니다."

    return {
        "summary": _one_line_summary(data.get("summary")),
        "detail": detail,
        "findings": findings,
        "strengths": strengths,
        "market_note": _clean_text(data.get("market_note")),
        "raw_text": raw_text,
    }


def render_diagnosis_markdown(diagnosis: dict[str, Any]) -> str:
    """기존 UI/PDF와 호환되는 읽기 좋은 마크다운."""
    lines = [f"### 한 줄 총평\n{diagnosis.get('summary', '')}"]

    detail = diagnosis.get("detail")
    if detail:
        lines.append(f"\n### 상세 분석\n{detail}")

    findings = diagnosis.get("findings", [])
    lines.append("\n### 발견된 문제")
    if not findings:
        lines.append("주요 개선 필요 항목이 발견되지 않았습니다.")
    else:
        for idx, f in enumerate(findings, 1):
            lines.append(
                f"{idx}. **{f.get('area', '검토 항목')} ({f.get('severity', 'low')})**\n"
                f"   - **어디가:** {f.get('what', '')}\n"
                f"   - **왜:** {f.get('why', '')}\n"
                f"   - **어떻게:** {f.get('how', '')}"
            )
            if f.get("market_basis"):
                lines.append(f"   - **시장 근거:** {f['market_basis']}")

    lines.append("\n### 장점")
    strengths = diagnosis.get("strengths", [])
    if strengths:
        lines.extend([f"- {s}" for s in strengths])
    else:
        lines.append("- 이미지에서 확인 가능한 별도의 장점을 찾지 못했습니다.")

    if diagnosis.get("market_note"):
        lines.append(f"\n### 시장 비교 참고\n{diagnosis['market_note']}")
    return "\n".join(lines)


def diagnose(file_bytes, media_type, api_key):
    """기존 app.py와의 호환을 위한 구형 진단 함수."""
    detailed = diagnose_detailed(file_bytes, media_type, api_key)
    return render_diagnosis_markdown(detailed)
