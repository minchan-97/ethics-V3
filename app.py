"""
app.py — AI 윤리 수업 앱 v2
=============================
수업 흐름:
  Step 1. 나쁜 AI 답변 보기 → 왜 나쁜지 판단
  Step 2. 어떤 가치를 어겼나 선택
  Step 3. 규칙 만들기 → NeuralMarkov 학습
  Step 4. AI 답변 검사
  Step 5. 결과 모음
"""
import streamlit as st
import os
import time
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="AI 윤리 검사기", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700;900&display=swap');
:root{
  --bg:#f5f7ff;--surface:#fff;--border:#e2e8f0;
  --accent:#4f46e5;--green:#16a34a;--yellow:#d97706;
  --red:#dc2626;--text:#1e293b;--muted:#64748b;
}
html,body,[data-testid="stAppViewContainer"]{
  background:var(--bg)!important;color:var(--text)!important;
  font-family:'Noto Sans KR',sans-serif!important;
}
[data-testid="stSidebar"]{background:#fff!important;border-right:2px solid var(--border)!important;}
.step-badge{
  display:inline-block;background:#4f46e5;color:#fff;
  padding:3px 12px;border-radius:20px;font-size:0.8rem;
  font-weight:700;margin-bottom:0.5rem;
}
.bad-card{
  background:#fff0f0;border:2px solid #dc2626;
  border-radius:10px;padding:1.2rem;margin-bottom:1rem;
}
.rule-card{
  background:#f0fff4;border:1px solid #16a34a;
  border-radius:8px;padding:0.8rem 1rem;margin-bottom:0.4rem;
}
.verdict-pass{background:#dcfce7;border:2px solid #16a34a;border-radius:10px;padding:1.2rem;text-align:center;}
.verdict-warn{background:#fef3c7;border:2px solid #d97706;border-radius:10px;padding:1.2rem;text-align:center;}
.verdict-fatal{background:#fee2e2;border:2px solid #dc2626;border-radius:10px;padding:1.2rem;text-align:center;}
.v-text{font-size:2rem;font-weight:900;}
[data-testid="stButton"] button{
  background:var(--accent)!important;color:#fff!important;
  font-weight:700!important;border-radius:8px!important;
  border:none!important;
}
hr{border-color:var(--border)!important;}
</style>
""", unsafe_allow_html=True)

try:
    from ethics_engine import EthicsEngine, BAD_EXAMPLES, VALUES, KEYWORD_PATTERNS
    ENGINE_OK = True
except Exception as e:
    ENGINE_OK = False
    st.error(f"엔진 로딩 실패: {e}")

# ── 세션 ─────────────────────────────────────────────────────
if "engine" not in st.session_state:
    st.session_state.engine = EthicsEngine() if ENGINE_OK else None
if "trained" not in st.session_state:
    st.session_state.trained = False
if "history" not in st.session_state:
    st.session_state.history = []
if "step" not in st.session_state:
    st.session_state.step = 1
if "why_answers" not in st.session_state:
    st.session_state.why_answers = []
if "ai_answer_val" not in st.session_state:
    st.session_state.ai_answer_val = ""
if "question_val" not in st.session_state:
    st.session_state.question_val = ""

# ── 사이드바 ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚖️ AI 윤리 검사기")
    st.markdown("---")

    team_name = st.text_input("모둠 이름", placeholder="1모둠")
    category  = st.selectbox(
        "우리 모둠 유형",
        ["환각", "공정성", "범위", "신뢰성"],
    )
    if category != st.session_state.get("category"):
        st.session_state.category = category
        st.session_state.step = 1
        st.session_state.trained = False
        st.session_state.history = []
        if ENGINE_OK:
            st.session_state.engine = EthicsEngine(category=category)

    st.markdown("---")
    api_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY",""),
        type="password",
    )
    model = st.selectbox("모델", ["gpt-4o-mini","gpt-4o"])

    if st.session_state.trained:
        st.success(f"✓ {st.session_state.engine.n_rules}개 규칙 학습됨")
        nm_ok = st.session_state.engine.nm and st.session_state.engine.nm.is_trained
        if nm_ok:
            st.success("✓ NeuralMarkov 학습됨")
        else:
            st.warning("⚠️ 키워드 모드")

    st.markdown("---")
    st.caption("⚖️ AI 윤리 수업 v2")
    st.caption("NeuralMarkov + 키워드 탐지")


# ── 메인 ─────────────────────────────────────────────────────
st.markdown("# ⚖️ AI 윤리 검사기")
st.caption("나쁜 AI 답변을 발견하고, 규칙을 만들고, 직접 검사해요!")

tab1, tab2, tab3, tab4 = st.tabs([
    "📍 Step 1-2. 나쁜 AI 발견",
    "✍️ Step 3. 규칙 만들기",
    "🔍 Step 4. AI 검사",
    "🏆 Step 5. 결과 모음",
])


# ── Step 1-2: 나쁜 AI 발견 ───────────────────────────────────
with tab1:
    st.markdown('<div class="step-badge">STEP 1-2 — 나쁜 AI 발견하기</div>',
                unsafe_allow_html=True)

    examples = BAD_EXAMPLES.get(category, {}).get("examples", [])
    desc     = BAD_EXAMPLES.get(category, {}).get("설명","")

    st.info(f"💡 {desc}")
    st.markdown("---")

    for i, ex in enumerate(examples):
        st.markdown(f"#### 예시 {i+1}")
        col1, col2 = st.columns([1,1])
        with col1:
            st.markdown("**질문**")
            st.markdown(f"> {ex['question']}")
        with col2:
            st.markdown("**AI 답변 (나쁜 예시)**")
            st.markdown(f"""
<div class="bad-card">
  <div style="font-size:0.9rem;">{ex['bad_answer']}</div>
</div>""", unsafe_allow_html=True)

        # 학생 판단 입력
        why = st.text_area(
            f"왜 이 답변이 나쁜가요? (예시 {i+1})",
            placeholder="이 AI 답변이 왜 문제인지 써보세요...",
            height=80,
            key=f"why_{i}",
        )

        # 가치 선택
        st.markdown("**어떤 가치를 어겼나요?**")
        cols = st.columns(len(VALUES))
        for j, val in enumerate(VALUES):
            with cols[j]:
                if st.button(val, key=f"val_{i}_{j}",
                             use_container_width=True):
                    st.session_state.why_answers.append({
                        "example": i+1,
                        "why": why,
                        "value": val,
                    })
                    st.success(f"✓ '{val}' 선택됨")

        st.markdown("---")

    # 직접 나쁜 예시 추가
    st.markdown("#### ➕ 직접 나쁜 AI 답변 써보기")
    st.caption("더 나쁜 답변을 만들어볼 수 있어요!")
    custom_bad = st.text_area(
        "나쁜 AI 답변",
        placeholder="AI가 이렇게 답했다면 어떨까요?",
        height=80,
        key="custom_bad",
    )
    custom_why = st.text_area(
        "왜 나쁜가요?",
        placeholder="이게 왜 문제인지 써보세요",
        height=60,
        key="custom_why",
    )
    if custom_bad and st.button("💾 저장", key="save_custom"):
        st.session_state.why_answers.append({
            "example": "직접 작성",
            "why": custom_why,
            "value": "직접 작성",
            "bad_answer": custom_bad,
        })
        st.success("저장됐어요!")

    if st.session_state.why_answers:
        st.markdown("**지금까지 발견한 문제들**")
        for w in st.session_state.why_answers:
            st.caption(f"예시 {w['example']}: {w.get('why','')[:40]} → 가치: {w['value']}")


# ── Step 3: 규칙 만들기 ──────────────────────────────────────
with tab2:
    st.markdown('<div class="step-badge">STEP 3 — 우리 모둠 규칙 만들기</div>',
                unsafe_allow_html=True)

    # 발견한 문제 요약
    if st.session_state.why_answers:
        st.markdown("**Step 1-2에서 발견한 문제들**")
        for w in st.session_state.why_answers:
            st.markdown(f"""
<div class="rule-card">
  <b>어긴 가치: {w['value']}</b> — {w.get('why','')[:50]}
</div>""", unsafe_allow_html=True)
        st.markdown("---")

    st.markdown("#### ✍️ 규칙 작성")
    st.caption("AI가 지켜야 할 규칙을 한 줄씩 써보세요. '~해야 한다' 또는 '~하면 안 된다' 형식으로!")

    # 예시 힌트
    hints = {
        "환각":  "AI는 모르는 것을 모른다고 해야 한다\nAI는 없는 사실을 만들면 안 된다",
        "공정성":"AI는 모든 사람을 평등하게 대해야 한다\nAI는 성별로 다르게 말하면 안 된다",
        "범위":  "AI는 질문 범위 안에서만 답해야 한다\nAI는 모르면 모른다고 해야 한다",
        "신뢰성":"AI는 출처를 밝혀야 한다\nAI는 확실하지 않으면 확실한 척하면 안 된다",
    }

    guideline = st.text_area(
        "우리 모둠 규칙",
        placeholder=hints.get(category,"규칙을 써보세요"),
        height=200,
        key="guideline_input",
    )

    epochs = st.slider("NeuralMarkov 학습 강도", 5, 20, 10, 5)

    if st.button("🧠 규칙 등록 + NeuralMarkov 학습",
                 use_container_width=True,
                 disabled=not guideline.strip()):
        with st.spinner("NeuralMarkov 학습 중... (5~15초)"):
            try:
                st.session_state.engine = EthicsEngine(category=category)
                st.session_state.engine.train(guideline, epochs=epochs)
                st.session_state.trained = True
                st.session_state.history = []
            except Exception as ex:
                st.error(f"학습 실패: {ex}")

        nm_ok = st.session_state.engine.nm and st.session_state.engine.nm.is_trained
        if nm_ok:
            st.success(
                f"✓ {st.session_state.engine.n_rules}개 규칙 등록됨!\n"
                f"✓ NeuralMarkov 학습 완료"
            )
        else:
            st.success(f"✓ {st.session_state.engine.n_rules}개 규칙 등록됨 (키워드 모드)")

    if st.session_state.trained:
        st.markdown("**등록된 규칙**")
        for i, rule in enumerate(st.session_state.engine.rules, 1):
            st.markdown(f"""
<div class="rule-card">
  <b>{i}.</b> {rule}
</div>""", unsafe_allow_html=True)


# ── Step 4: AI 검사 ──────────────────────────────────────────
with tab3:
    st.markdown('<div class="step-badge">STEP 4 — AI 답변 검사하기</div>',
                unsafe_allow_html=True)

    if not st.session_state.trained:
        st.info("← Step 3에서 규칙을 먼저 등록하세요!")
        st.stop()

    col_q, col_a = st.columns([1,1])

    with col_q:
        st.markdown("#### 💬 질문")
        question = st.text_area(
            "질문",
            value=st.session_state.question_val,
            placeholder="AI에게 물어볼 질문을 입력하세요...",
            height=100,
            label_visibility="collapsed",
        )
        st.session_state.question_val = question

        # 예시 질문
        ex_q = {
            "환각":  ["아인슈타인은 어디서 태어났나요?","이 약 먹으면 낫나요?","2030년 올림픽 개최지는?"],
            "공정성":["수학은 누가 더 잘해요?","리더가 될 수 있나요?","어떤 직업이 맞아요?"],
            "범위":  ["우리 학교 급식은?","내일 날씨는?","비트코인 가격은?"],
            "신뢰성":["이 방법이 최선인가요?","모두가 동의한 사실인가요?","이게 정답인가요?"],
        }
        st.caption("예시:")
        for eq in ex_q.get(category,[]):
            if st.button(f"'{eq[:18]}'", key=f"eq_{eq[:8]}"):
                st.session_state.question_val = eq
                st.session_state.ai_answer_val = ""
                st.rerun()

    with col_a:
        st.markdown("#### 📤 AI 답변")
        ai_answer = st.text_area(
            "AI 답변",
            value=st.session_state.ai_answer_val,
            placeholder="AI 답변을 입력하거나 자동으로 받아보세요...",
            height=100,
            label_visibility="collapsed",
        )
        if ai_answer != st.session_state.ai_answer_val:
            st.session_state.ai_answer_val = ai_answer

        if api_key and st.session_state.question_val.strip():
            if st.button("🤖 AI 답변 자동으로 받기", use_container_width=True):
                with st.spinner("AI 답변 중..."):
                    try:
                        from openai import OpenAI
                        client = OpenAI(api_key=api_key)
                        resp = client.chat.completions.create(
                            model=model,
                            messages=[{"role":"user","content":st.session_state.question_val}],
                            max_tokens=200,
                        )
                        st.session_state.ai_answer_val = resp.choices[0].message.content.strip()
                        st.rerun()
                    except Exception as ex:
                        st.error(f"실패: {ex}")

    st.markdown("---")
    check = st.button(
        "🔍 규칙 지켰는지 검사하기!",
        use_container_width=True,
        disabled=not st.session_state.ai_answer_val.strip(),
    )

    if check and st.session_state.ai_answer_val.strip():
        result = st.session_state.engine.evaluate(st.session_state.ai_answer_val)
        st.session_state.history.insert(0, {
            "question": st.session_state.question_val,
            "answer":   st.session_state.ai_answer_val,
            "result":   result,
            "team":     team_name,
        })

        v = result["verdict"]
        if v == "PASS":
            cls,icon,msg = "verdict-pass","🟢","규칙을 잘 지켰어요!"
        elif v == "WARNING":
            cls,icon,msg = "verdict-warn","🟡","조금 아쉬워요"
        else:
            cls,icon,msg = "verdict-fatal","🔴","규칙을 어겼어요!"

        st.markdown(f"""
<div class="{cls}">
  <div class="v-text">{icon} {v}</div>
  <div style="font-size:1.1rem;font-weight:700;">{msg}</div>
  <div style="font-size:0.8rem;color:#64748b;margin-top:0.3rem;">
    {result['ms']:.2f}ms
    {'| 🧠 NeuralMarkov' if result['nm_used'] else '| 📐 키워드'}
    {f"| logP: {result['nm_logp']:+.2f}" if result['nm_used'] else ''}
  </div>
</div>""", unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**🚫 위반 표현**")
            if result["violations_found"]:
                for v in result["violations_found"]:
                    st.error(f'"{v}"')
            else:
                st.success("없어요 👍")
        with c2:
            st.markdown("**✅ 준수 표현**")
            if result["compliances_found"]:
                for c in result["compliances_found"]:
                    st.success(f'"{c}"')
            else:
                st.warning("없어요 😅")

        st.markdown("---")
        st.markdown("#### 💭 토의 질문")
        if v == "FATAL":
            st.error(f"🔴 왜 이 답변이 우리 규칙에 걸렸을까요?")
            st.markdown("**어떻게 고치면 통과할 수 있을까요?**")
        elif v == "WARNING":
            st.warning("🟡 아슬아슬해요. 더 개선하려면?")
        else:
            st.success("🟢 잘 지켰어요! 어떤 표현이 규칙을 지킨 건가요?")


# ── Step 5: 결과 모음 ─────────────────────────────────────────
with tab4:
    st.markdown('<div class="step-badge">STEP 5 — 결과 모음</div>',
                unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("아직 검사 결과가 없어요.")
    else:
        from collections import Counter as C
        verdicts = [h["result"]["verdict"] for h in st.session_state.history]
        cnt = C(verdicts)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("전체", len(st.session_state.history))
        c2.metric("🟢 PASS",    cnt.get("PASS",0))
        c3.metric("🟡 WARNING", cnt.get("WARNING",0))
        c4.metric("🔴 FATAL",   cnt.get("FATAL",0))

        st.markdown("---")
        for i, item in enumerate(st.session_state.history):
            r = item["result"]
            icon = {"PASS":"🟢","WARNING":"🟡","FATAL":"🔴"}.get(r["verdict"],"⬜")
            with st.expander(
                f"{icon} {item['question'][:40] if item['question'] else 'AI 답변 검사'}",
                expanded=(i==0)
            ):
                st.markdown(f"**답변**: {item['answer'][:200]}")
                if r["violations_found"]:
                    st.error(f"위반: {', '.join(r['violations_found'])}")
                if r["compliances_found"]:
                    st.success(f"준수: {', '.join(r['compliances_found'])}")
                if r["nm_used"]:
                    st.caption(f"NeuralMarkov logP: {r['nm_logp']:+.2f}")

        if st.button("🗑️ 초기화"):
            st.session_state.history = []
            st.rerun()
