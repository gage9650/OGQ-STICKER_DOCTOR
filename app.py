# app.py — OGQ 스티커 닥터
# 기존 VER2 기능 유지 + OGQ 시장 비교 + 심사 기준 커스텀 + AI 문제 위치 표시
# 시장 비교 AI가 실패해도 앱 전체가 중단되지 않도록 별도 오류 처리와 모델 fallback을 사용한다.

from __future__ import annotations

import hashlib
import time

import streamlit as st
from PIL import Image, ImageDraw

from checker import SPECS, check_image
from diagnosis import diagnose_detailed, render_diagnosis_markdown
from criteria import DEFAULT_CRITERIA, PLATFORM_PRESETS, build_selected_criteria
from market_analysis import build_market_analysis_prompt
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
    """AI가 반환한 0~1000 bbox를 실제 이미지 좌표로 변환해 빨간 타원으로 표시한다."""
    import io

    image = Image.open(io.BytesIO(file_bytes)).convert("RGBA")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for idx, finding in enumerate(findings, 1):
        bbox = finding.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        left, top, right, bottom = bbox
        box = (
            int(left / 1000 * width),
            int(top / 1000 * height),
            int(right / 1000 * width),
            int(bottom / 1000 * height),
        )
        draw.ellipse(
            box,
            outline="#D92D20",
            width=max(3, min(width, height) // 80),
        )
        draw.text(
            (box[0] + 4, max(0, box[1] - 22)),
            str(idx),
            fill="#D92D20",
        )

    return image


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
    *,
    max_output_tokens: int = 1400,
) -> tuple[str, str | None]:
    """시장 비교 AI를 호출한다.

    3.8 → 3.7 → 3.6 순으로 ServerError가 발생했을 때만 fallback한다.
    AI 실패 시에도 OGQ 검색 결과 자체는 계속 사용할 수 있도록 예외를 문자열로 돌려준다.
    """
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError, ServerError

    client = genai.Client(api_key=gemini_key)
    models = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"]
    last_error: Exception | None = None

    for model_name in models:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=market_prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_output_tokens,
                    ),
                )
                text = (response.text or "").strip()
                if text:
                    return text, None
                return "", f"{model_name}이 빈 응답을 반환했습니다."
            except ServerError as exc:
                last_error = exc
                if attempt == 0:
                    time.sleep(1.0)
                    continue
                break
            except APIError as exc:
                last_error = exc
                # 5xx 계열만 다음 모델로 fallback. 인증/쿼터/요청 오류는 숨기지 않는다.
                code = getattr(exc, "code", None)
                if isinstance(code, int) and 500 <= code < 600:
                    break
                return "", f"Gemini API 오류({code or '알 수 없음'}): {str(exc)}"
            except Exception as exc:
                last_error = exc
                break

    detail = str(last_error) if last_error else "알 수 없는 오류"
    return "", f"Gemini 시장 비교 분석에 실패했습니다. 검색 결과는 정상적으로 표시할 수 있습니다. ({detail})"


# ---------- 0단계: 검사 기준 ----------
st.header("검사 기준 설정")
st.caption("OGQ 공개 가이드를 기본으로 불러오고, 원하는 검사 항목만 선택하거나 나만의 기준을 추가할 수 있어요.")

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

custom_rule_text = st.text_input(
    "나만의 검사 기준 추가 (선택)",
    placeholder="예: 캐릭터 얼굴이 이미지의 30% 이상 보이는지 확인",
)
custom_rules = [custom_rule_text] if custom_rule_text.strip() else []
selected_criteria = build_selected_criteria(
    "내가 직접 선택",
    selected_criteria,
    custom_rules,
)

with st.expander("현재 선택한 검사 기준 보기"):
    for criterion in selected_criteria:
        st.write(f"✓ {criterion}")
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

    if st.button("🔎 OGQ 시장과 비교하기", type="primary"):
        # 이전 결과를 먼저 비워서 검색 실패 시 오래된 결과가 보이지 않도록 한다.
        st.session_state.pop("market_results", None)
        st.session_state.pop("market_analysis", None)
        st.session_state.pop("market_prompt", None)
        st.session_state.pop("market_analysis_error", None)

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

                    st.session_state["market_results"] = market_results

                    if market_results:
                        market_prompt = build_market_analysis_prompt(
                            feelings,
                            user_tags,
                            market_results[:8],
                        )
                        st.session_state["market_prompt"] = market_prompt

                        gemini_key = _get_secret("GEMINI_API_KEY")
                        if gemini_key:
                            with st.spinner("OGQ 시장 결과를 AI가 비교 분석하고 있어요..."):
                                analysis, error_message = _generate_market_ai(
                                    gemini_key,
                                    market_prompt,
                                    max_output_tokens=1400,
                                )
                            if analysis:
                                st.session_state["market_analysis"] = analysis
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

        if st.session_state.get("market_analysis"):
            st.subheader("AI 시장 비교")
            st.markdown(st.session_state["market_analysis"])

        if st.session_state.get("market_analysis_error"):
            st.warning(st.session_state["market_analysis_error"])

    # ---------- 4단계: AI 진단 + 위치 표시 ----------
    st.divider()
    st.header("4단계 · AI가 어디가 문제인지 표시")
    st.caption("선택한 검사 영역과 시장 비교 자료를 바탕으로 문제 영역을 표시하고, 무엇을/왜/어떻게 고칠지 설명합니다.")

    market_context = st.session_state.get("market_prompt", "")

    if st.button(f"🧠 업로드한 {len(files)}개 전체 AI 진단 받기", type="primary"):
        gemini_key = _get_secret("GEMINI_API_KEY")
        if not gemini_key:
            st.error("Gemini API 키가 설정되지 않았어요. Streamlit Secrets에 GEMINI_API_KEY를 추가해주세요.")
        else:
            progress = st.progress(0, text="AI가 스티커를 살펴보는 중...")
            for i, file_result in enumerate(all_file_results):
                digest = hashlib.sha256(file_result["bytes"]).hexdigest()[:16]
                criteria_key = hashlib.md5(
                    ",".join(selected_criteria).encode("utf-8")
                ).hexdigest()[:8]
                market_key = hashlib.md5(
                    market_context.encode("utf-8")
                ).hexdigest()[:8]
                cache_key = f"diag_v4_{file_result['name']}_{digest}_{criteria_key}_{market_key}"

                if cache_key not in st.session_state:
                    try:
                        detailed = diagnose_detailed(
                            file_result["bytes"],
                            file_result["mime"],
                            gemini_key,
                            selected_criteria=selected_criteria,
                            market_context=market_context,
                        )
                        st.session_state[cache_key] = detailed
                    except Exception as exc:
                        st.session_state[cache_key] = {
                            "summary": f"진단 중 오류가 발생했어요: {exc}",
                            "findings": [],
                            "strengths": [],
                            "market_note": "",
                            "raw_text": "",
                        }

                progress.progress(
                    (i + 1) / len(all_file_results),
                    text=f"{i + 1}/{len(all_file_results)} 완료",
                )
            progress.empty()
            st.success("전체 AI 진단이 끝났어요.")

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
