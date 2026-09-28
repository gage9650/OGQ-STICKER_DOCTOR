# app.py — OGQ 스티커 닥터
# 기존 VER2 기능 유지 + OGQ 시장 비교 + 심사 기준 커스텀 + AI 문제 위치 표시
# 시장 비교 AI가 실패해도 앱 전체가 중단되지 않도록 별도 오류 처리와 모델 fallback을 사용한다.

from __future__ import annotations

import hashlib
import json
import time

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw

from checker import SPECS, check_image
from diagnosis import diagnose_detailed, render_diagnosis_markdown
from criteria import DEFAULT_CRITERIA, PLATFORM_PRESETS, build_selected_criteria
from market_analysis import (
    MARKET_SCHEMA,
    build_market_analysis_prompt,
    build_market_context_for_diagnosis,
    download_market_reference_images,
    normalize_market_analysis,
    generate_public_web_check,
)
from ogq_market import OGQAPIError, search_by_keywords
from report_utils import (
    build_pdf_report,
    build_priority_todo,
    compute_score,
    load_history,
    save_history_entry,
)


st.set_page_config(
    page_title="OGQ 스티커 닥터",
    page_icon="🩺",
    layout="centered",
)


st.markdown(
    """
    <style>
    :root {
        --mint-primary: #0F9B8E;
        --mint-light: #EEFBF9;
        --mint-border: #BFEFE9;
    }
    div[data-testid="stExpander"] {
        border: 1px solid var(--mint-border);
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(15, 155, 142, 0.08);
        overflow: hidden;
    }
    div[data-testid="stExpander"] summary { font-weight: 600; }
    button[kind="secondary"], button[kind="primary"] {
        border-radius: 999px !important;
        font-weight: 600 !important;
    }
    div[data-testid="stFileUploaderDropzone"] {
        border-radius: 14px;
        border: 1.5px dashed var(--mint-border);
        background-color: var(--mint-light);
    }
    .hero-banner {
        background: linear-gradient(135deg, #0F9B8E 0%, #14B8A6 100%);
        border-radius: 18px;
        padding: 28px 32px;
        color: white;
        margin-bottom: 28px;
    }
    .hero-banner h1 { margin: 0 0 6px 0; font-size: 1.6rem; }
    .hero-banner p { margin: 0; opacity: 0.92; font-size: 0.95rem; }
    .score-card {
        background: var(--mint-light);
        border: 1px solid var(--mint-border);
        border-radius: 16px;
        padding: 18px 22px;
        margin-bottom: 20px;
    }
    .score-card .score-num {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--mint-primary);
    }
    .todo-row {
        border-left: 4px solid var(--mint-primary);
        background: white;
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 8px;
    }
    .todo-row.fail { border-left-color: #B3261E; }
    .todo-row.warn { border-left-color: #8A6300; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-banner">
        <h1>🩺 OGQ 스티커 닥터</h1>
        <p>내 스티커의 문제를 찾고, OGQ 시장의 기존 콘텐츠와 비교해 수정 방향을 잡아보세요.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def _draw_annotations(file_bytes: bytes, findings: list[dict]) -> Image.Image:
    """Gemini 공식 bbox 형식 [ymin, xmin, ymax, xmax]을 실제 이미지 좌표로 변환한다."""
    import io

    image = Image.open(io.BytesIO(file_bytes)).convert("RGBA")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for idx, finding in enumerate(findings, 1):
        bbox = finding.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        try:
            ymin, xmin, ymax, xmax = [max(0, min(1000, float(v))) for v in bbox]
        except (TypeError, ValueError):
            continue
        if xmax <= xmin or ymax <= ymin:
            continue

        # Gemini 공식 형식: [ymin, xmin, ymax, xmax]
        left = int(xmin / 1000 * width)
        top = int(ymin / 1000 * height)
        right = int(xmax / 1000 * width)
        bottom = int(ymax / 1000 * height)

        if right <= left or bottom <= top:
            continue

        # 지나치게 큰 영역은 표시하지 않는다. (모델이 전체 화면을 잘못 지정한 경우)
        area_ratio = ((right - left) * (bottom - top)) / max(1, width * height)
        if area_ratio > 0.90:
            continue

        line_width = max(3, min(width, height) // 85)
        draw.ellipse(
            (left, top, right, bottom),
            outline="#D92D20",
            width=line_width,
        )
        badge_left = max(0, left)
        badge_top = max(0, top - 28)
        draw.rounded_rectangle(
            (badge_left, badge_top, badge_left + 26, badge_top + 24),
            radius=8,
            fill="#D92D20",
        )
        draw.text(
            (badge_left + 8, badge_top + 4),
            str(idx),
            fill="#FFFFFF",
        )

    return image


def _speak_completion(message: str = "") -> None:
    """진단 완료 시 브라우저에서 짧은 2음 벨소리(띠링)를 재생한다.

    기존 음성 합성(TTS)은 사용하지 않는다. 브라우저의 자동재생 정책에 의해
    소리가 차단될 경우를 대비해 수동 재생 버튼을 함께 제공한다.
    """
    components.html(
        """
        <div style="font-family:sans-serif; padding:2px 0;">
          <button id="ding-btn" style="border:1px solid #BFEFE9;border-radius:999px;padding:7px 14px;background:#EEFBF9;color:#0B3B36;font-weight:600;cursor:pointer;">🔔 완료음 다시 듣기</button>
        </div>
        <script>
        (() => {
          let played = false;

          function ding() {
            try {
              const AudioContext = window.AudioContext || window.webkitAudioContext;
              if (!AudioContext) return;

              const ctx = new AudioContext();
              const now = ctx.currentTime;

              const gain = ctx.createGain();
              gain.gain.setValueAtTime(0.0001, now);
              gain.gain.exponentialRampToValueAtTime(0.22, now + 0.015);
              gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.42);
              gain.connect(ctx.destination);

              const osc1 = ctx.createOscillator();
              osc1.type = 'sine';
              osc1.frequency.setValueAtTime(880, now);
              osc1.frequency.exponentialRampToValueAtTime(1320, now + 0.10);
              osc1.connect(gain);

              const osc2 = ctx.createOscillator();
              osc2.type = 'sine';
              osc2.frequency.setValueAtTime(1320, now + 0.12);
              osc2.frequency.exponentialRampToValueAtTime(1760, now + 0.22);
              osc2.connect(gain);

              osc1.start(now);
              osc1.stop(now + 0.11);
              osc2.start(now + 0.12);
              osc2.stop(now + 0.28);

              setTimeout(() => { try { ctx.close(); } catch (_) {} }, 650);
              played = true;
            } catch (_) {
              // 브라우저 자동재생/AudioContext 제한 시 수동 버튼으로 재생 가능
            }
          }

          const btn = document.getElementById('ding-btn');
          btn?.addEventListener('click', ding);
          setTimeout(() => { if (!played) ding(); }, 100);
        })();
        </script>
        """,
        height=45,
    )


def _get_secret(name: str, default: str = "") -> str:
    """Streamlit Secrets에서 문자열 값을 안전하게 읽는다."""
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    return str(value or "").strip()


def _keywords_from_user_input(feelings: str, user_tags: list[str], limit: int = 5) -> list[str]:
    """느낌/태그 입력을 OGQ 검색용 키워드로 정리하고 중복을 제거한다."""
    raw_values: list[str] = []

    for part in (feelings or "").replace("\n", ",").split(","):
        value = " ".join(part.strip().split())
        if value:
            raw_values.append(value)

    for tag in user_tags:
        value = " ".join(str(tag or "").strip().lstrip("#").split())
        if value:
            raw_values.append(value)

    result: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
        if len(result) >= max(1, limit):
            break

    return result


def _generate_market_ai(
    gemini_key: str,
    market_prompt: str,
    user_image_bytes: bytes,
    user_image_mime: str,
    market_results: list[dict],
    *,
    max_output_tokens: int = 1700,
) -> tuple[dict, str | None]:
    """사용자 실제 이미지와 OGQ 참조 이미지를 함께 비교 분석한다."""
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError, ServerError
    import json
    import random
    import time

    client = genai.Client(api_key=gemini_key)
    models = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"]
    refs = download_market_reference_images(market_results[:4])

    visual_instruction = """
[이미지 비교 규칙]
- 첫 번째 첨부 이미지는 '사용자 스티커'입니다.
- 이후 첨부 이미지는 각각 'OGQ 검색 결과 [번호]'의 실제 이미지입니다.
- 사용자가 입력한 태그는 보조 정보일 뿐이며, 이미지 자체보다 우선하지 않습니다.
- 사용자의 태그만 보고 시장과의 차이를 만들어내지 마세요.
- 사용자 이미지와 시장 참조 이미지가 실제로 같은 스티커이거나 거의 동일하면 그 사실을 명확하게 인정하고 유사성을 매우 높음으로 평가하세요.
- 검색 결과의 제목/태그가 서로 달라도 실제 이미지가 같다면 '다르다'고 쓰지 마세요.
- differences는 실제 이미지 또는 제목/설명/태그 중 확인 가능한 근거가 있을 때만 작성하세요.
- comparisons는 실제로 비교 가능한 결과만 최대 4개 작성하세요.
"""
    contents = [
        market_prompt + "\n" + visual_instruction,
        "[USER STICKER IMAGE]",
        types.Part.from_bytes(data=user_image_bytes, mime_type=user_image_mime),
    ]
    for ref in refs:
        contents.extend([
            f"[OGQ SEARCH RESULT {ref['index']}] {ref['title']}",
            types.Part.from_bytes(data=ref["bytes"], mime_type=ref["mime_type"]),
        ])

    last_error: Exception | None = None
    for model_name in models:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=MARKET_SCHEMA,
                        max_output_tokens=max_output_tokens,
                        temperature=0.15,
                    ),
                )
                raw = (response.text or "").strip()
                if not raw:
                    raise RuntimeError(f"{model_name}이 빈 응답을 반환했습니다.")
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    start_json = raw.find("{")
                    end_json = raw.rfind("}")
                    if start_json < 0 or end_json <= start_json:
                        raise RuntimeError("시장 분석 결과가 JSON으로 반환되지 않았습니다.")
                    parsed = json.loads(raw[start_json:end_json + 1])
                return normalize_market_analysis(parsed), None
            except (ServerError, APIError, RuntimeError, json.JSONDecodeError) as exc:
                last_error = exc
                code = getattr(exc, "code", None)
                transient = isinstance(exc, ServerError) or code in {408, 429, 500, 502, 503, 504}
                if transient and attempt < 1:
                    time.sleep(min(6.0, 1.5 * (2 ** attempt)) + random.uniform(0, 0.4))
                    continue
                if not transient:
                    break

    return {}, (
        "Gemini 시장 비교 분석에 실패했습니다. 검색된 OGQ 콘텐츠는 정상적으로 확인할 수 있습니다. "
        f"({last_error})"
    )


def _run_ai_diagnostics(
    all_file_results: list[dict],
    selected_criteria: list[str],
    custom_rules: list[str],
    market_context: str,
    gemini_key: str,
) -> bool:
    """전체 스티커 AI 진단을 실행한다. 성공/실패와 무관하게 UI가 계속 렌더링되도록 결과를 세션에 저장한다."""
    progress = st.progress(0, text="AI가 스티커를 살펴보는 중...")
    had_success = False
    for i, file_result in enumerate(all_file_results):
        digest = hashlib.sha256(file_result["bytes"]).hexdigest()[:16]
        criteria_key = hashlib.md5(",".join(selected_criteria).encode("utf-8")).hexdigest()[:8]
        market_key = hashlib.md5(market_context.encode("utf-8")).hexdigest()[:8]
        cache_key = f"diag_v5_{file_result['name']}_{digest}_{criteria_key}_{market_key}"

        if cache_key not in st.session_state:
            try:
                detailed = diagnose_detailed(
                    file_result["bytes"],
                    file_result["mime"],
                    gemini_key,
                    selected_criteria=selected_criteria,
                    market_context=market_context,
                    custom_criteria=custom_rules,
                )
                st.session_state[cache_key] = detailed
                if detailed.get("model_used"):
                    had_success = True
            except Exception as exc:
                st.session_state[cache_key] = {
                    "summary": "이번 스티커의 AI 진단은 일시적으로 완료되지 않았습니다.",
                    "detail": "Gemini 서버가 일시적으로 바쁘거나 요청을 처리하지 못했습니다. 자동 재시도와 대체 모델을 시도했지만 응답을 받지 못했습니다. 규격 검사 결과는 계속 확인할 수 있습니다.",
                    "findings": [],
                    "strengths": [],
                    "market_note": "",
                    "custom_criteria_results": [],
                    "raw_text": "",
                    "model_used": "",
                    "diagnosis_error": str(exc),
                }
        else:
            had_success = True

        progress.progress((i + 1) / len(all_file_results), text=f"{i + 1}/{len(all_file_results)} 완료")

    progress.empty()
    return had_success


# ---------- 0단계: 검사 기준 ----------
st.header("검사 기준 설정")
st.caption("OGQ 공개 가이드를 기본으로 불러오고, 원하는 검사 항목만 선택하거나 나만의 기준을 원하는 만큼 추가할 수 있어요.")

preset_names = list(PLATFORM_PRESETS.keys()) + ["내가 직접 선택"]
preset = st.selectbox("검사 기준 프로필", preset_names, index=0)

default_names = [name for name, _ in DEFAULT_CRITERIA]
if preset == "내가 직접 선택":
    selected_default = default_names
else:
    selected_default = PLATFORM_PRESETS.get(preset, default_names)

selected_criteria = st.multiselect(
    "검사할 심사 영역",
    options=default_names,
    default=selected_default,
)

if "custom_rules" not in st.session_state:
    st.session_state["custom_rules"] = []

st.markdown("**나만의 검사 기준**")
custom_rule_text = st.text_input(
    "새 검사 기준",
    key="custom_rule_input",
    placeholder="예: 캐릭터 얼굴이 이미지의 30% 이상 보이는지 확인",
)
if st.button("＋ 검사 기준 추가"):
    rule = " ".join(custom_rule_text.strip().split())
    if not rule:
        st.warning("추가할 검사 기준을 입력해주세요.")
    elif rule in st.session_state["custom_rules"]:
        st.info("이미 추가된 검사 기준입니다.")
    else:
        st.session_state["custom_rules"].append(rule)
        st.success("검사 기준이 추가됐습니다. 다음 AI 진단에 적용됩니다.")

custom_rules = list(st.session_state["custom_rules"])
if custom_rules:
    for idx, rule in enumerate(custom_rules):
        c1, c2 = st.columns([8, 1])
        with c1:
            st.write(f"**{idx + 1}.** {rule}")
        with c2:
            if st.button("삭제", key=f"delete_custom_rule_{idx}"):
                st.session_state["custom_rules"].pop(idx)
                st.rerun()
else:
    st.caption("아직 사용자 지정 기준이 없습니다. 원하는 만큼 추가할 수 있습니다.")

selected_criteria = build_selected_criteria("내가 직접 선택", selected_criteria, custom_rules)
standard_selected = [item for item in selected_criteria if item not in custom_rules]
with st.expander("이번 AI 진단에 적용되는 기준 확인", expanded=True):
    st.success(f"적용됨 · 기본/공개 기준 {len(standard_selected)}개 + 나만의 기준 {len(custom_rules)}개")
    if standard_selected:
        st.markdown("**기본/공개 기준**")
        for criterion in standard_selected:
            st.write(f"✓ {criterion}")
    if custom_rules:
        st.markdown("**나만의 기준 — 개별 결과가 별도 표시됩니다**")
        for rule in custom_rules:
            st.write(f"✓ {rule}")

st.caption("파일 해상도·용량·형식 같은 기술 규격 검사는 기본으로 유지되고, 위에서 선택한 영역은 AI 심층 진단에 적용됩니다.")

# ---------- 1단계: 이미지 + 사용자 설명 ----------
st.header("1단계 · 스티커 정보 입력")
feelings = st.text_input(
    "이 스티커는 어떤 느낌인가요?",
    placeholder="예: 귀여움, 장난스러움, 직장인 공감, 살짝 시니컬함",
)
tag_text = st.text_input(
    "태그를 입력해주세요",
    placeholder="쉼표로 구분해서 입력: 강아지, 직장인, 출근, 피곤",
)
user_tags = [t.strip().lstrip("#") for t in tag_text.split(",") if t.strip()]

st.header("2단계 · 스티커 업로드")
st.caption("OGQ 공개 제작 가이드 기준: 메인 240x240 · 스티커 740x640 · 탭 96x74, 각 1MB 이하, RGB, 투명 배경")

files = st.file_uploader(
    "스티커 이미지를 올려주세요 (여러 장 가능)",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)

if files:
    type_counts = {name: 0 for name in SPECS}
    all_file_results = []

    for f in files:
        file_bytes = f.getvalue()
        img_type, results = check_image(file_bytes, f.name)
        if img_type:
            type_counts[img_type] += 1

        all_file_results.append(
            {
                "name": f.name,
                "bytes": file_bytes,
                "mime": f.type or "image/png",
                "img_type": img_type,
                "results": results,
            }
        )

        fail_count = sum(1 for grade, _, _ in results if grade == "fail")
        icon = "❌" if fail_count else "✅"
        with st.expander(
            f"{icon} {f.name} — 문제 {fail_count}건",
            expanded=fail_count > 0,
        ):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(file_bytes)
            with col2:
                for grade, item, msg in results:
                    if grade == "pass":
                        st.success(f"**{item}** — {msg}")
                    elif grade == "warn":
                        st.warning(f"**{item}** — {msg}")
                    else:
                        st.error(f"**{item}** — {msg}")

    # ---------- 3단계: 시장 비교 ----------
    st.divider()
    st.header("3단계 · OGQ 시장 비교")
    st.caption("입력한 느낌과 태그를 키워드로 OGQ 마켓의 관련 스티커를 찾고, AI가 공통점·차이점·장단점을 분석합니다.")

    auto_diagnose_after_market = st.checkbox(
        "시장 비교가 끝나면 바로 AI 문제 표시까지 실행",
        value=False,
        key="auto_diagnose_after_market",
        help="체크하면 시장 비교와 시장 분석이 끝난 직후 선택한 검사 기준으로 AI 진단을 자동 실행합니다.",
    )

    if st.button("🔎 OGQ 시장과 비교하기", type="primary"):
        # 이전 결과를 먼저 비워서 검색 실패 시 오래된 결과가 보이지 않도록 한다.
        st.session_state.pop("market_results", None)
        st.session_state.pop("market_analysis", None)
        st.session_state.pop("market_context", None)
        st.session_state.pop("market_analysis_error", None)
        st.session_state.pop("public_web_check", None)
        st.session_state.pop("public_web_check_error", None)
        st.session_state.pop("auto_diag_completed_for_market_key", None)

        ogq_api_key = _get_secret("OGQ_API_KEY")
        base_url = _get_secret(
            "OGQ_API_BASE_URL",
            "https://4th-ai-ogq.competition.ogq.me",
        )

        if not ogq_api_key:
            st.error("OGQ API 키가 설정되지 않았어요. Streamlit Secrets에 OGQ_API_KEY를 추가해주세요.")
        else:
            keywords = _keywords_from_user_input(feelings, user_tags, limit=5)
            if not keywords:
                st.warning("시장 검색에 사용할 느낌이나 태그를 하나 이상 입력해주세요.")
            else:
                try:
                    with st.spinner("OGQ 마켓을 검색하고 있어요..."):
                        market_results = search_by_keywords(
                            ogq_api_key,
                            keywords,
                            max_keywords=5,
                            per_keyword=8,
                            max_results=12,
                            max_detail_requests=8,
                            base_url=base_url,
                            user_id=None,
                        )

                    # 업로드 이미지와 OGQ 검색 결과의 강한 시각 일치 신호를 먼저 계산
                    from market_analysis import annotate_visual_match_hints
                    market_results = annotate_visual_match_hints(
                        all_file_results[0]["bytes"], market_results
                    )
                    # 강한 일치 신호가 있는 항목을 앞쪽으로 우선 배치
                    market_results.sort(
                        key=lambda item: float(item.get("visual_match_score") or 0.0),
                        reverse=True,
                    )
                    st.session_state["market_results"] = market_results

                    if market_results:
                        market_prompt = build_market_analysis_prompt(
                            feelings,
                            user_tags,
                            market_results[:8],
                        )

                        gemini_key = _get_secret("GEMINI_API_KEY")
                        if gemini_key:
                            with st.spinner("OGQ 시장 결과를 실제 이미지와 비교 분석하고 있어요..."):
                                analysis, error_message = _generate_market_ai(
                                    gemini_key,
                                    market_prompt,
                                    all_file_results[0]["bytes"],
                                    all_file_results[0]["mime"],
                                    market_results[:8],
                                    max_output_tokens=1700,
                                )
                            if analysis:
                                st.session_state["market_analysis"] = analysis

                                # OGQ 데이터와 별개로, 공개 웹에서 대중적 유사성도 보조 조사한다.
                                web_check = generate_public_web_check(
                                    gemini_key,
                                    feelings,
                                    user_tags,
                                )
                                if web_check.get("text"):
                                    st.session_state["public_web_check"] = web_check
                                elif web_check.get("error"):
                                    st.session_state["public_web_check_error"] = web_check["error"]

                                # 이미지 진단에는 실제 시장 분석 + 공개 웹 보조 조사 결과를 전달한다.
                                st.session_state["market_context"] = build_market_context_for_diagnosis(
                                    analysis,
                                    market_results[:8],
                                    st.session_state.get("public_web_check", {}),
                                )
                            if error_message:
                                st.session_state["market_analysis_error"] = error_message
                        else:
                            st.session_state["market_analysis_error"] = (
                                "Gemini API 키가 없어 시장 검색 결과만 표시합니다."
                            )
                    else:
                        st.session_state["market_analysis_error"] = (
                            "관련 OGQ 콘텐츠를 찾지 못했습니다. 다른 느낌이나 태그를 입력해 보세요."
                        )

                except OGQAPIError as exc:
                    st.error(f"OGQ 시장 검색에 실패했어요: {exc}")

    # 시장 비교 완료 직후 자동 진단 옵션이 켜져 있으면 바로 AI 문제 표시까지 진행
    if auto_diagnose_after_market and st.session_state.get("market_results") and st.session_state.get("market_context"):
        if not st.session_state.get("auto_diag_completed_for_market_key"):
            auto_key = hashlib.md5(st.session_state.get("market_context", "").encode("utf-8")).hexdigest()[:12]
            gemini_key = _get_secret("GEMINI_API_KEY")
            if gemini_key:
                with st.spinner("시장 비교가 끝났어요. 바로 AI가 문제 위치를 분석하고 있어요..."):
                    _run_ai_diagnostics(
                        all_file_results,
                        selected_criteria,
                        custom_rules,
                        st.session_state.get("market_context", ""),
                        gemini_key,
                    )
                st.session_state["auto_diag_completed_for_market_key"] = auto_key
                st.success("시장 비교에 이어 AI 문제 표시까지 완료했어요.")
                _speak_completion("시장 비교와 AI 문제 진단이 모두 완료되었습니다.")

    market_results = st.session_state.get("market_results", [])
    if market_results:
        st.subheader(f"관련 콘텐츠 {len(market_results)}개")
        cols = st.columns(4)
        for i, item in enumerate(market_results[:8]):
            with cols[i % 4]:
                if item.get("main_image_url"):
                    st.image(item["main_image_url"], use_container_width=True)
                st.caption(item.get("title", "제목 없음"))
                if item.get("description"):
                    st.caption(item["description"][:90])
                if item.get("matched_keyword"):
                    st.caption(f"검색어: {item['matched_keyword']}")
                tags = item.get("tags") or []
                if tags:
                    st.caption("태그: " + ", ".join(f"#{tag}" for tag in tags[:6]))

        market_analysis = st.session_state.get("market_analysis")
        if isinstance(market_analysis, dict):
            st.subheader("AI 시장 비교")

            level = market_analysis.get("similarity_level", "보통")
            level_emoji = {"높음": "🔴", "보통": "🟡", "낮음": "🟢"}.get(level, "🟡")
            st.markdown(f"**시장 유사성: {level_emoji} {level}**")
            if market_analysis.get("similarity_summary"):
                st.write(market_analysis["similarity_summary"])

            comparisons = market_analysis.get("comparisons") or []
            if comparisons:
                st.markdown("**시장 콘텐츠별 유사성**")
                for comp in comparisons:
                    try:
                        idx = int(comp.get("result_index", 0))
                    except (TypeError, ValueError):
                        continue
                    if not (1 <= idx <= len(market_results[:8])):
                        continue
                    source = market_results[idx - 1]
                    st.markdown(f"**[{idx}] {source.get('title', 'OGQ 콘텐츠')} — 유사성: {comp.get('similarity_level', '보통')}**")
                    if comp.get("similarity_reason"):
                        st.write(comp["similarity_reason"])
                    if comp.get("same_points"):
                        st.caption("같은 점: " + " · ".join(comp["same_points"]))
                    if comp.get("different_points"):
                        st.caption("다른 점: " + " · ".join(comp["different_points"]))

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**시장 전반의 공통 요소**")
                similarities = market_analysis.get("similarities") or []
                if similarities:
                    for item in similarities:
                        st.markdown(f"- {item}")
                else:
                    st.caption("검색 결과에서 반복적으로 확인되는 공통 요소가 없습니다.")
            with c2:
                st.markdown("**시장과의 차이점**")
                differences = market_analysis.get("differences") or []
                if differences:
                    for item in differences:
                        st.markdown(f"- {item}")
                else:
                    st.caption("검색 데이터만으로 확인되는 뚜렷한 차이점이 없습니다.")

            st.markdown("**시장과 비교했을 때의 장점**")
            for item in (market_analysis.get("strengths") or []):
                st.markdown(f"- {item}")
            if not market_analysis.get("strengths"):
                st.caption("검색 결과만으로 판단할 수 있는 장점을 찾지 못했습니다.")

            st.markdown("**보완하면 좋은 점**")
            for item in (market_analysis.get("gaps") or []):
                st.markdown(f"- {item}")
            if not market_analysis.get("gaps"):
                st.caption("검색 결과만으로 확인되는 보완점이 없습니다.")

            priority = market_analysis.get("priority_actions") or []
            if priority:
                st.markdown("**먼저 확인할 것**")
                for idx, item in enumerate(priority, 1):
                    st.markdown(f"{idx}. {item}")

            evidence = market_analysis.get("evidence") or []
            if evidence:
                with st.expander("시장 분석 근거 보기"):
                    for item in evidence:
                        idx = item.get("result_index")
                        reason = item.get("reason", "")
                        if idx and 1 <= int(idx) <= len(market_results[:8]):
                            source = market_results[int(idx) - 1]
                            st.markdown(
                                f"**[{idx}] {source.get('title', '콘텐츠')}** — {reason}"
                            )

        elif st.session_state.get("market_analysis"):
            # 이전 버전의 문자열 결과가 세션에 남아도 화면이 깨지지 않도록 호환
            st.subheader("AI 시장 비교")
            st.markdown(str(st.session_state["market_analysis"]))

        public_web = st.session_state.get("public_web_check")
        if isinstance(public_web, dict) and public_web.get("text"):
            st.markdown("**공개 웹 대중적 유사성 참고**")
            st.caption("OGQ 마켓 밖의 공개 웹 자료를 보조적으로 확인한 결과입니다. 법적 표절·저작권 판단이 아닙니다.")
            st.markdown(public_web["text"])
            sources = public_web.get("sources") or []
            if sources:
                with st.expander("웹 검색 근거 보기"):
                    for source in sources:
                        title = source.get("title") or source.get("uri")
                        uri = source.get("uri") or ""
                        st.markdown(f"- [{title}]({uri})")
        if st.session_state.get("public_web_check_error"):
            st.caption(st.session_state["public_web_check_error"])

        if st.session_state.get("market_analysis_error"):
            st.warning(st.session_state["market_analysis_error"])

    # ---------- 4단계: AI 진단 + 위치 표시 ----------
    st.divider()
    st.header("4단계 · AI가 어디가 문제인지 표시")
    st.caption("선택한 검사 영역과 시장 비교 자료를 바탕으로 문제 영역을 표시하고, 무엇을/왜/어떻게 고칠지 설명합니다.")

    market_context = st.session_state.get("market_context", "")

    if st.button(f"🧠 업로드한 {len(files)}개 전체 AI 진단 받기", type="primary"):
        gemini_key = _get_secret("GEMINI_API_KEY")
        if not gemini_key:
            st.error("Gemini API 키가 설정되지 않았어요. Streamlit Secrets에 GEMINI_API_KEY를 추가해주세요.")
        else:
            completed = _run_ai_diagnostics(
                all_file_results,
                selected_criteria,
                custom_rules,
                market_context,
                gemini_key,
            )
            if completed:
                st.success("전체 AI 진단이 끝났어요.")
            else:
                st.warning("AI 진단이 완료되지 않은 파일이 있습니다. 잠시 후 다시 시도해 주세요.")
            _speak_completion()


    # 결과 렌더링
    diagnosis_texts: dict[str, str] = {}
    for file_result in all_file_results:
        digest = hashlib.sha256(file_result["bytes"]).hexdigest()[:16]
        cache_prefix = f"diag_v4_{file_result['name']}_{digest}_"
        matched_keys = [
            key
            for key in st.session_state.keys()
            if isinstance(key, str) and key.startswith(cache_prefix)
        ]
        if not matched_keys:
            continue

        diagnosis = st.session_state[matched_keys[-1]]
        diagnosis_texts[file_result["name"]] = render_diagnosis_markdown(diagnosis)

        with st.expander(
            f"🧠 AI 진단 — {file_result['name']}",
            expanded=True,
        ):
            st.markdown(diagnosis_texts[file_result["name"]])

            if diagnosis.get("diagnosis_error"):
                st.warning("⚠️ 이번 AI 진단 요청에서 일시적인 오류가 발생했습니다. 아래 진단 결과와 시장 자료가 있다면 계속 참고할 수 있습니다.")

            if custom_rules:
                custom_results = diagnosis.get("custom_criteria_results") or []
                st.subheader("나만의 검사 기준 결과")
                returned = {str(item.get("criterion", "")).strip() for item in custom_results if isinstance(item, dict)}
                st.success(f"사용자 지정 기준 {len(custom_rules)}개가 이번 AI 요청에 포함됐고, {len(returned)}/{len(custom_rules)}개 개별 평가가 반환되었습니다.")
                status_icon = {"양호": "🟢", "주의": "🟡", "개선 필요": "🔴", "판단 어려움": "⚪"}
                for rule in custom_rules:
                    item = next((x for x in custom_results if isinstance(x, dict) and str(x.get("criterion", "")).strip() == rule), None)
                    if not item:
                        item = {"status": "판단 어려움", "result": "개별 평가가 반환되지 않았습니다.", "evidence": "응답 누락"}
                    st.markdown(f"**{status_icon.get(item.get('status'), '⚪')} {rule} · {item.get('status', '판단 어려움')}**")
                    st.write(item.get("result", ""))
                    if item.get("evidence"):
                        st.caption(f"근거: {item['evidence']}")

            findings = diagnosis.get("findings", [])
            if findings:
                st.subheader("문제 위치")
                annotated = _draw_annotations(file_result["bytes"], findings)
                st.image(
                    annotated,
                    caption="🔴 AI가 문제 위치로 판단한 영역 — 참고용 시각화",
                    use_container_width=True,
                )

                for idx, finding in enumerate(findings, 1):
                    severity_label = {
                        "high": "🔴 높음",
                        "medium": "🟡 중간",
                        "low": "🟢 낮음",
                    }.get(finding.get("severity"), "검토")
                    st.markdown(
                        f"**{idx}. {severity_label} · {finding.get('area', '검토 항목')}**\n\n"
                        f"- **어디가:** {finding.get('what', '')}\n"
                        f"- **왜:** {finding.get('why', '')}\n"
                        f"- **어떻게:** {finding.get('how', '')}"
                    )
            else:
                st.info("이미지에서 위치를 특정할 수 있는 개선 항목이 없습니다.")

    # ---------- 5단계: 기존 셀프 체크리스트 ----------
    st.divider()
    st.header("5단계 · 규정 위반 셀프 체크리스트")
    st.caption("이미지만으로 확정하기 어려운 항목은 직접 확인해 주세요.")

    checklist_items = {
        "저작권 있는 폰트를 상업적으로 이용 가능한 라이선스로만 사용했다": "font_license",
        "생성형 AI 사용 여부와 관련 규정을 확인했다": "ai_rule_check",
        "다른 판매자의 기존 콘텐츠와 차별화되는 요소가 있다": "no_duplicate",
        "텍스트가 잘리지 않고 세이프존 안에 들어와 있다": "text_safezone",
        "욕설·폭력·선정성·정치/종교 관련 부적합 요소가 없다": "no_sensitive_content",
    }
    checklist_status = {}
    for label, key in checklist_items.items():
        checklist_status[label] = st.checkbox(label, key=f"chk_{key}")

    checklist_done = sum(1 for value in checklist_status.values() if value)
    checklist_total = len(checklist_items)
    st.caption(f"체크리스트 {checklist_done}/{checklist_total} 완료")

    # ---------- 6단계: 점수 + Todo ----------
    st.divider()
    st.header("6단계 · 준비도 & 개선 우선순위")
    score = compute_score(
        all_file_results,
        checklist_done,
        checklist_total,
    )
    st.markdown(
        f"""
        <div class="score-card">
            <div>OGQ 준비도 점수</div>
            <div class="score-num">{score:.0f}점 <span style="font-size:1rem; font-weight:400;">/ 100점</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    todo_list = build_priority_todo(all_file_results)
    if todo_list:
        st.subheader("이것부터 고치세요")
        for i, todo in enumerate(todo_list, start=1):
            grade_label = "❌ 실패" if todo["grade"] == "fail" else "⚠️ 주의"
            css_class = "fail" if todo["grade"] == "fail" else "warn"
            st.markdown(
                f"""
                <div class="todo-row {css_class}">
                    <b>{i}. {grade_label} · {todo['item']}</b><br/>
                    {todo['msg']}<br/>
                    <small>영향받은 파일 {todo['affected_count']}개: {', '.join(todo['affected_files'][:5])}
                    {' 외' if len(todo['affected_files']) > 5 else ''}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.success("자동 검사 기준으로 고칠 항목이 없어요!")

    # ---------- 7단계: PDF + 히스토리 ----------
    st.divider()
    st.header("7단계 · 리포트 내보내기 & 재검사 히스토리")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("📄 PDF 리포트 생성"):
            pdf_bytes = build_pdf_report(
                all_file_results,
                todo_list,
                score,
                checklist_status,
                diagnosis_texts,
            )
            st.download_button(
                "⬇️ PDF 다운로드",
                data=pdf_bytes,
                file_name="ogq_sticker_doctor_report.pdf",
                mime="application/pdf",
            )

    with col_b:
        if st.button("💾 이번 결과를 히스토리에 저장"):
            pass_count = sum(
                1
                for fr in all_file_results
                for grade, _, _ in fr["results"]
                if grade == "pass"
            )
            warn_count = sum(
                1
                for fr in all_file_results
                for grade, _, _ in fr["results"]
                if grade == "warn"
            )
            fail_count = sum(
                1
                for fr in all_file_results
                for grade, _, _ in fr["results"]
                if grade == "fail"
            )
            save_history_entry(
                score,
                pass_count,
                warn_count,
                fail_count,
                checklist_done,
                checklist_total,
            )
            st.success("히스토리에 저장했어요.")

    history = load_history()
    if len(history) >= 2:
        st.subheader("재검사 히스토리")
        st.line_chart({"score": [h["score"] for h in history]})
        st.caption(
            f"최근 {len(history)}회 검사 · 마지막: {history[-1]['timestamp']} ({history[-1]['score']}점)"
        )
    elif len(history) == 1:
        st.caption("아직 히스토리가 1개예요. 수정 후 다시 저장하면 변화 그래프가 나타납니다.")

    # ---------- 제출 구성 요약 ----------
    st.divider()
    st.subheader("제출 구성 체크")
    st.caption("OGQ 공개 제작 가이드: 메인 1개 · 스티커 24개 · 탭 1개")
    need = {"메인 이미지": 1, "스티커 이미지": 24, "탭 이미지": 1}
    for name, required in need.items():
        have = type_counts[name]
        st.write(f"**{name}**  {have} / {required}")
        st.progress(min(have / required, 1.0))
else:
    st.info("이미지를 올리면 OGQ 심사 기준에 맞는지 바로 검사합니다.")
