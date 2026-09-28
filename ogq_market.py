"""OGQ Market API 연동.
기본 엔드포인트: https://oapi.ogq.me
API Key는 코드에 넣지 않고 Streamlit Secrets에서 읽는다.
"""
from __future__ import annotations

import requests

DEFAULT_BASE_URL = "https://4th-ai-ogq.competition.ogq.me"


class OGQAPIError(RuntimeError):
    """OGQ API 요청 실패."""


def search_stickers(
    api_key: str,
    keyword: str = "",
    page: int = 0,
    page_size: int = 10,
    user_id: str = "sticker-doctor",
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = 12,
) -> dict:
    """OGQ 스티커 검색 API 호출.

    공식 문서 기준:
    GET /v1/stickers
    Header: X-OAPI-KEY
    Query: userId, pageSize, page, keyword
    """
    if not api_key:
        raise OGQAPIError("OGQ API 키가 설정되지 않았습니다.")

    page_size = max(1, min(int(page_size), 100))
    params = {
        "userId": user_id,
        "pageSize": page_size,
        "page": max(0, int(page)),
    }
    if keyword.strip():
        params["keyword"] = keyword.strip()

    url = f"{base_url.rstrip('/')}/v1/stickers"
    try:
        response = requests.get(
            url,
            params=params,
            headers={"X-OAPI-KEY": api_key},
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise OGQAPIError(f"OGQ API 요청에 실패했습니다: {exc}") from exc
    except ValueError as exc:
        raise OGQAPIError("OGQ API 응답을 JSON으로 읽지 못했습니다.") from exc

    if isinstance(data, dict) and data.get("code") not in (None, 20000, 200):
        raise OGQAPIError(f"OGQ API가 오류를 반환했습니다. code={data.get('code')}")

    return data


def parse_sticker_items(payload: dict) -> list[dict]:
    """OGQ API의 검색 결과에서 화면/AI 분석에 필요한 필드만 추린다."""
    data = payload.get("data") if isinstance(payload, dict) else {}
    elements = data.get("elements", []) if isinstance(data, dict) else []

    parsed = []
    for element in elements:
        if not isinstance(element, dict):
            continue
        # API 문서의 일반적인 검색 응답은 element 자체가 sticker 정보다.
        sticker = element.get("sticker") if isinstance(element.get("sticker"), dict) else element
        if not isinstance(sticker, dict):
            continue

        parsed.append(
            {
                "content_id": sticker.get("contentId", ""),
                "title": sticker.get("title", "제목 없음"),
                "description": sticker.get("description", ""),
                "main_image_url": sticker.get("mainImageUrl", ""),
                "tab_image_url": sticker.get("tabImageUrl", ""),
                "animated": bool(sticker.get("animated", False)),
                "creator_name": (sticker.get("creator") or {}).get("name", "알 수 없음"),
                "matched_keyword": element.get("_matched_keyword", ""),
            }
        )

    return parsed


def search_by_keywords(
    api_key: str,
    keywords: list[str],
    *,
    max_keywords: int = 5,
    per_keyword: int = 8,
    base_url: str = DEFAULT_BASE_URL,
    user_id: str = "sticker-doctor",
) -> list[dict]:
    """여러 키워드로 검색하되 중복 콘텐츠는 제거한다.

    API 호출량을 제한하기 위해 기본 최대 5개 키워드만 검색한다.
    """
    unique_keywords = []
    seen = set()
    for raw in keywords:
        kw = " ".join(str(raw).strip().split())
        if not kw:
            continue
        low = kw.casefold()
        if low in seen:
            continue
        seen.add(low)
        unique_keywords.append(kw)
        if len(unique_keywords) >= max_keywords:
            break

    if not unique_keywords:
        unique_keywords = [""]

    merged: dict[str, dict] = {}
    for kw in unique_keywords:
        payload = search_stickers(
            api_key,
            keyword=kw,
            page=0,
            page_size=per_keyword,
            user_id=user_id,
            base_url=base_url,
        )
        for item in parse_sticker_items(payload):
            item["_matched_keyword"] = kw
            cid = item["content_id"] or f"{item['title']}::{item['main_image_url']}"
            if cid not in merged:
                merged[cid] = item

    return list(merged.values())
