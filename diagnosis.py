# diagnosis.py — Gemini AI 스티커 진단
from __future__ import annotations

import json
import re
from typing import Any

from google import genai
from google.genai import types


SYSTEM_PROMPT = """당신은 스티커 콘텐츠를 검토하는 AI 진단 도구입니다.
공식적으로 공개된 OGQ 제작 가이드와 사용자가 선택한 검사 기준을 바탕으로 분석하세요.

중요:
- OGQ의 비공개 내부 심사 매뉴얼을 알고 있다고 주장하지 마세요.
- '심사 통과 확률'을 예측하지 마세요.
- 이미지에서 실제로 확인할 수 없는 사실은 지어내지 마세요.
- 문제를 발견하면 가능한 한 정확히 이미지 안의 위치를 추정해 bbox로 표시하세요.
- bbox 좌표는 이미지의 실제 픽셀 좌표가 아니라 0~1000 정규화 좌표입니다.
- bbox는 [left, top, right, bottom] 순서이고 0~1000 범위입니다.
- 위치를 특정하기 어려우면 bbox를 빈 배열로 두세요.
- 수정안은 '무엇을, 왜, 어떻게'가 드러나는 구체적인 수준으로 작성하세요.
"""


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def _normalise_bbox(bbox: Any) -> list[int]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return []
    try:
        values = [max(0, min(1000, int(float(v)))) for v in bbox]
        if values[2] <= values[0] or values[3] <= values[1]:
            return []
        return values
    except (TypeError, ValueError):
        return []


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

[시장 비교 자료]
{market_context or "시장 비교를 실행하지 않았습니다."}

다음 JSON 객체 하나만 반환하세요.

{{
  "summary": "한 줄 총평",
  "findings": [
    {{
      "severity": "high|medium|low",
      "area": "문제 영역 이름",
      "what": "무엇이 문제인지",
      "why": "왜 문제인지. 시장 비교 자료가 있으면 시장에서 확인된 패턴과 연결",
      "how": "구체적인 수정 기획안",
      "bbox": [left, top, right, bottom]
    }}
  ],
  "strengths": ["구체적인 장점 1", "구체적인 장점 2"],
  "market_note": "시장 비교 자료가 있을 경우 참고용으로 해석한 차이점. 없으면 빈 문자열"
}}

findings에는 실제로 개선할 가치가 있는 항목만 넣으세요. 문제 위치를 특정하기 어려운 항목은 bbox를 []로 두세요.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[
            types.Part.from_bytes(data=file_bytes, mime_type=media_type),
            user_prompt,
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            max_output_tokens=2500,
        ),
    )
    data = _extract_json(response.text)
    if not data:
        return {
            "summary": response.text or "AI 진단 결과를 읽지 못했습니다.",
            "findings": [],
            "strengths": [],
            "market_note": "",
            "raw_text": response.text or "",
        }

    findings = []
    for item in data.get("findings", []):
        if not isinstance(item, dict):
            continue
        findings.append(
            {
                "severity": item.get("severity", "low"),
                "area": item.get("area", "검토 항목"),
                "what": item.get("what", ""),
                "why": item.get("why", ""),
                "how": item.get("how", ""),
                "bbox": _normalise_bbox(item.get("bbox")),
            }
        )

    data["findings"] = findings
    data.setdefault("strengths", [])
    data.setdefault("summary", "")
    data.setdefault("market_note", "")
    data["raw_text"] = response.text or ""
    return data


def render_diagnosis_markdown(diagnosis: dict[str, Any]) -> str:
    """기존 UI/PDF와도 호환되는 읽기 좋은 마크다운으로 변환."""
    lines = [f"### 한 줄 총평\n{diagnosis.get('summary', '')}"]

    findings = diagnosis.get("findings", [])
    lines.append("\n### 발견된 문제")
    if not findings:
        lines.append("주요 개선 필요 항목이 발견되지 않았습니다.")
    else:
        for idx, f in enumerate(findings, 1):
            lines.append(
                f"{idx}. **{f.get('area', '검토 항목')} ({f.get('severity', 'low')})**\n"
                f"   - 무엇이 문제인가: {f.get('what', '')}\n"
                f"   - 왜 확인해야 하는가: {f.get('why', '')}\n"
                f"   - 어떻게 바꿀 것인가: {f.get('how', '')}"
            )

    lines.append("\n### 장점")
    strengths = diagnosis.get("strengths", [])
    if strengths:
        lines.extend([f"- {s}" for s in strengths])
    else:
        lines.append("- 특별히 입력된 장점이 없습니다.")

    if diagnosis.get("market_note"):
        lines.append(f"\n### 시장 비교 참고\n{diagnosis['market_note']}")
    return "\n".join(lines)


def diagnose(file_bytes, media_type, api_key):
    """기존 app.py와의 호환을 위한 구형 진단 함수."""
    detailed = diagnose_detailed(file_bytes, media_type, api_key)
    return render_diagnosis_markdown(detailed)
