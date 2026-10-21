"""Korean single-candidate swing scanner. Research only; never places orders."""
from __future__ import annotations

from datetime import date, datetime, timedelta
import math
import threading
import time
import uuid
from typing import Any
from zoneinfo import ZoneInfo

import FinanceDataReader as fdr
import pandas as pd
import requests
import streamlit as st

APP_TITLE = "알려줘 · 주식 스윙 분석기"
# Keep the task registry across Streamlit reruns. Jobs can still stop if the hosting process sleeps/restarts.
@st.cache_resource
def get_task_store():
    return {}, threading.Lock()

TASKS, TASKS_LOCK = get_task_store()
DEFAULT_CAPITAL = 300_000
RISK_FRACTION = 0.01
MAX_DATA_AGE_DAYS = 5
MIN_HISTORY_ROWS = 60
MIN_TURNOVER_WON = 1_000_000_000
MIN_TURNOVER_RATIO = 1.5
MIN_VOLUME_RATIO = 1.2
MAX_DAILY_CHANGE_PCT = 8.0

st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="centered")
st.title("📈 주식 스윙 분석기")
st.caption("국내 주식 · 규칙 기반 분석 · 자동 주문 없음")
st.warning(
    "교육·조사용 도구입니다. 수익을 보장하지 않으며, 데이터 지연·가격 급변·갭 하락·수수료·세금 때문에 "
    "실제 결과가 계산과 달라질 수 있습니다. 주문 전 최신 호가와 공시를 직접 확인하세요."
)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


@st.cache_data(ttl=900, show_spinner=False)
def get_listing(market: str) -> pd.DataFrame:
    """Return a cleaned listing. Raises a useful error if the provider format changes."""
    df = fdr.StockListing(market)
    if df is None or df.empty:
        raise RuntimeError(f"{market} 상장 종목 목록이 비어 있습니다. 데이터 제공처 응답을 확인하세요.")
    df = df.copy()
    # FinanceDataReader versions/providers differ: KRX listings often use Code,
    # while other listing formats use Symbol. Normalize both to Symbol.
    code_col = next((c for c in ("Symbol", "Code", "종목코드", "단축코드") if c in df.columns), None)
    name_col = next((c for c in ("Name", "종목명", "회사명") if c in df.columns), None)
    if code_col is None or name_col is None:
        raise RuntimeError(
            "상장 목록의 열 이름이 예상과 다릅니다. "
            f"받은 열: {', '.join(map(str, df.columns[:15]))}"
        )
    if code_col != "Symbol":
        df["Symbol"] = df[code_col]
    if name_col != "Name":
        df["Name"] = df[name_col]
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
    df["Name"] = df["Name"].astype(str).str.strip()
    df = df[df["Symbol"].str.fullmatch(r"\d{6}", na=False)]
    df = df[(df["Name"] != "") & (df["Name"].str.lower() != "nan")]
    df = df.drop_duplicates(subset=["Symbol"])
    if "Marcap" in df.columns:
        df["Marcap"] = pd.to_numeric(df["Marcap"], errors="coerce")
        df = df.sort_values("Marcap", ascending=False, na_position="last")
    return df.reset_index(drop=True)


@st.cache_data(ttl=900, show_spinner=False)
def get_history(code: str, start: str, end_exclusive: str) -> pd.DataFrame:
    """Fetch daily OHLCV; normalize numeric columns and remove malformed rows."""
    df = fdr.DataReader(code, start, end_exclusive)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy().sort_index()
    required = {"High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["High", "Low", "Close", "Volume"])
    df = df[(df["Close"] > 0) & (df["High"] > 0) & (df["Low"] > 0) & (df["Volume"] >= 0)]
    # Duplicate timestamps can make rolling calculations ambiguous.
    return df[~df.index.duplicated(keep="last")]


def analyze_one(code: str, name: str, hist: pd.DataFrame, today: date) -> dict[str, Any] | None:
    """Calculate indicators, freshness, signal, risk levels and quantity for one ticker."""
    if hist is None or hist.empty or len(hist) < MIN_HISTORY_ROWS:
        return None

    d = hist.copy().sort_index()
    d["SMA20"] = d["Close"].rolling(20, min_periods=20).mean()
    d["SMA50"] = d["Close"].rolling(50, min_periods=50).mean()
    prev_close = d["Close"].shift(1)
    true_range = pd.concat(
        [d["High"] - d["Low"], (d["High"] - prev_close).abs(), (d["Low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    d["ATR14"] = true_range.rolling(14, min_periods=14).mean()
    d["Turnover"] = d["Close"] * d["Volume"]  # estimate, not official investor flow
    d["TurnoverAvg20"] = d["Turnover"].shift(1).rolling(20, min_periods=20).mean()
    d["VolAvg20"] = d["Volume"].shift(1).rolling(20, min_periods=20).mean()

    last = d.iloc[-1]
    last_date = pd.Timestamp(d.index[-1]).date()
    age_days = (today - last_date).days
    fresh = 0 <= age_days <= MAX_DATA_AGE_DAYS
    close = _as_float(last["Close"])
    atr = _as_float(last["ATR14"])
    sma20 = _as_float(last["SMA20"])
    sma50 = _as_float(last["SMA50"])
    turnover_avg = _as_float(last["TurnoverAvg20"])
    volume_avg = _as_float(last["VolAvg20"])
    turnover = _as_float(last["Turnover"])
    volume = _as_float(last["Volume"])
    previous_close = _as_float(d["Close"].iloc[-2])

    if not all(math.isfinite(x) and x > 0 for x in [close, atr, sma20, sma50, turnover, previous_close]):
        return None
    if not math.isfinite(turnover_avg) or turnover_avg <= 0 or not math.isfinite(volume_avg) or volume_avg <= 0:
        return None

    # Exclude the current row from the breakout reference to avoid look-ahead bias.
    prior_20 = d.iloc[-21:-1]
    if len(prior_20) != 20:
        return None
    prior_20_high = _as_float(prior_20["High"].max())
    if not math.isfinite(prior_20_high) or prior_20_high <= 0:
        return None

    daily_pct = (close / previous_close - 1.0) * 100.0
    turnover_ratio = turnover / turnover_avg
    volume_ratio = volume / volume_avg
    trend = close > sma20 > sma50
    breakout = close > prior_20_high
    liquid = turnover >= MIN_TURNOVER_WON
    momentum = (
        turnover_ratio >= MIN_TURNOVER_RATIO
        and volume_ratio >= MIN_VOLUME_RATIO
        and 0 <= daily_pct < MAX_DAILY_CHANGE_PCT
    )
    # 최종 진입 신호는 기존의 엄격한 조건을 그대로 유지한다.
    # 후보 점수는 검색 결과 정렬/표시용이며 매수 신호로 사용하지 않는다.
    daily_ok = 0 <= daily_pct < MAX_DAILY_CHANGE_PCT
    turnover_ratio_ok = turnover_ratio >= MIN_TURNOVER_RATIO
    volume_ratio_ok = volume_ratio >= MIN_VOLUME_RATIO
    signal = bool(
        fresh and trend and breakout and liquid and turnover_ratio_ok
        and volume_ratio_ok and daily_ok
    )
    candidate_checks = {
        "상승 추세": trend,
        "20일 고점 돌파": breakout,
        "추정 거래대금 10억 이상": liquid,
        "거래대금 증가 조건": turnover_ratio_ok,
        "거래량 증가 조건": volume_ratio_ok,
        "당일 상승률 조건": daily_ok,
        "데이터 최신성": fresh,
    }
    candidate_score = sum(bool(v) for v in candidate_checks.values())
    failed_checks = [label for label, ok in candidate_checks.items() if not ok]

    entry = close if breakout else prior_20_high
    stop = entry - 2.0 * atr
    target = entry + 3.0 * atr
    stop_gap = entry - stop
    capital = DEFAULT_CAPITAL
    risk_budget = capital * RISK_FRACTION
    if entry <= 0 or stop <= 0 or stop_gap <= 0:
        return None
    risk_shares = math.floor(risk_budget / stop_gap)
    capital_shares = math.floor(capital / entry)
    shares = min(risk_shares, capital_shares) if signal else 0

    return {
        "code": str(code).zfill(6), "name": name, "date": last_date.isoformat(),
        "age_days": age_days, "fresh": fresh, "close": close, "sma20": sma20, "sma50": sma50,
        "atr": atr, "prior_20_high": prior_20_high, "turnover": turnover,
        "turnover_ratio": turnover_ratio, "volume_ratio": volume_ratio, "daily_pct": daily_pct,
        "trend": trend, "breakout": breakout, "liquid": liquid, "momentum": momentum,
        "daily_ok": daily_ok, "turnover_ratio_ok": turnover_ratio_ok,
        "volume_ratio_ok": volume_ratio_ok, "candidate_score": candidate_score,
        "failed_checks": failed_checks, "signal": signal, "entry": entry, "stop": stop, "target": target,
        "risk_budget": risk_budget, "shares": max(0, shares),
        "estimated_cost": max(0, shares) * entry,
        "planned_risk": max(0, shares) * stop_gap,
    }


def send_telegram_notification(message: str) -> tuple[bool, str]:
    """Send an optional Telegram message when Streamlit secrets are configured."""
    try:
        token = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "")
    except Exception:
        token, chat_id = "", ""
    if not token or not chat_id:
        return False, "텔레그램 알림 미설정"
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": message}, timeout=12,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            return False, "텔레그램 API가 알림 전송을 확인하지 못함"
        return True, "텔레그램 알림 전송 완료"
    except Exception as exc:
        return False, f"텔레그램 알림 전송 실패: {type(exc).__name__}"


def background_scan(task_id: str, market: str, universe_size: int) -> None:
    """Run scanning away from the Streamlit page render loop."""
    def update_progress(value: float, message: str) -> None:
        with TASKS_LOCK:
            if task_id in TASKS:
                TASKS[task_id]["progress"] = max(0.0, min(1.0, float(value)))
                TASKS[task_id]["progress_text"] = message
    try:
        with TASKS_LOCK:
            TASKS[task_id]["status"] = "running"
            TASKS[task_id]["started_at"] = time.time()
        results, errors, diagnostics = scan_market(market, universe_size, update_progress)
        signal = next((item for item in results if item.get("signal")), None)
        if signal:
            summary = (f"주식 스캐너 조사 완료: 진입 조건을 충족한 후보 {signal['name']}({signal['code']}) 발견. "
                       f"기준일 {signal['date']}. 앱에서 가격과 조건을 직접 확인하세요. 자동 주문은 하지 않았습니다.")
        else:
            summary = f"주식 스캐너 조사 완료: {universe_size}종목을 확인했지만 기존 진입 조건을 모두 충족한 종목은 없습니다. 매수 신호 없음."
        with TASKS_LOCK:
            TASKS[task_id].update({"status":"done", "results":results, "errors":errors,
                                   "diagnostics":diagnostics, "finished_at":time.time(),
                                   "progress":1.0, "progress_text":"조사 완료", "summary":summary})
        ok, notify_msg = send_telegram_notification(summary)
        with TASKS_LOCK:
            TASKS[task_id]["notification"] = notify_msg
    except Exception as exc:
        summary = f"주식 스캐너 조사 실패: {type(exc).__name__}: {exc}"
        with TASKS_LOCK:
            if task_id in TASKS:
                TASKS[task_id].update({"status":"error", "errors":[summary],
                                       "finished_at":time.time(), "progress_text":"조사 실패", "summary":summary})
        send_telegram_notification(summary)


def scan_market(market: str, universe_size: int, progress_callback=None) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Scan the chosen universe. Returns valid rows, including stale rows marked ineligible, and error samples."""
    today = datetime.now(ZoneInfo("Asia/Seoul")).date()
    start = today - timedelta(days=180)
    end_exclusive = today + timedelta(days=1)
    listing = get_listing(market).head(universe_size)
    if listing.empty:
        raise RuntimeError("선택한 시장에서 분석할 종목을 찾지 못했습니다.")

    found: list[dict[str, Any]] = []
    errors: list[str] = []
    empty_history = 0
    short_history = 0
    total = len(listing)
    for i, row in enumerate(listing.itertuples(index=False), start=1):
        code = str(getattr(row, "Symbol")).zfill(6)
        name = str(getattr(row, "Name"))
        try:
            hist = get_history(code, start.isoformat(), end_exclusive.isoformat())
            if hist.empty:
                empty_history += 1
            elif len(hist) < MIN_HISTORY_ROWS:
                short_history += 1
            item = analyze_one(code, name, hist, today) if not hist.empty else None
            if item is not None:
                # Keep stale rows visible for diagnosis; signal still requires fresh=True.
                found.append(item)
        except Exception as exc:
            # Keep a few concise examples for troubleshooting without flooding the UI.
            if len(errors) < 5:
                errors.append(f"{name}({code}): {type(exc).__name__}")
        if i % 5 == 0 or i == total:
            if progress_callback is not None:
                progress_callback(i / total, f"종목 분석 중: {i}/{total}")
    found.sort(
        key=lambda x: (
            x["signal"], x["fresh"], x["candidate_score"], x["trend"], x["breakout"],
            x["turnover_ratio"], x["volume_ratio"], x["daily_pct"]
        ),
        reverse=True,
    )
    diagnostics = [f"종목 목록: {total}개", f"가격 데이터 없음: {empty_history}개", f"거래 이력 부족: {short_history}개"]
    return found, errors, diagnostics


with st.expander("분석 설정", expanded=True):
    market = st.selectbox("분석 시장", ["KRX", "KOSPI", "KOSDAQ"], index=0)
    universe_size = st.select_slider("분석 종목 수", options=[200, 500, 1000, 1500], value=500)
    st.caption("기본 검색 범위 500종목 · 최대 1,500종목까지 검색 · 투자금 300,000원 · 최종 진입 조건은 기존 기준 유지")

if "task_id" not in st.session_state:
    st.session_state.task_id = None
if "last_seen_finished_task" not in st.session_state:
    st.session_state.last_seen_finished_task = None

# Optional mobile push via Telegram. Tokens are kept in Streamlit Secrets, not in source code.
with st.expander("백그라운드 조사 및 휴대폰 알림 설정", expanded=False):
    st.write("조사는 서버의 별도 작업 스레드에서 진행됩니다. 앱 화면을 다른 앱으로 전환해도 서버가 살아 있는 동안은 계속될 수 있습니다.")
    st.write("휴대폰 알림을 받으려면 Telegram 봇을 만든 뒤 Streamlit Cloud의 App settings → Secrets에 TELEGRAM_BOT_TOKEN과 TELEGRAM_CHAT_ID를 설정해야 합니다.")
    st.caption("중요: 무료 Streamlit Cloud가 앱을 절전/재시작하면 백그라운드 작업이 중단될 수 있어 24시간 무중단 보장은 아닙니다.")

if st.button("알려줘", type="primary", use_container_width=True):
    existing_id = st.session_state.task_id
    with TASKS_LOCK:
        existing = TASKS.get(existing_id) if existing_id else None
    if existing and existing.get("status") == "running":
        st.warning("이미 조사가 진행 중이야. 완료될 때까지 기다려 줘.")
    else:
        task_id = str(uuid.uuid4())
        with TASKS_LOCK:
            TASKS[task_id] = {"status":"queued", "progress":0.0, "progress_text":"조사를 준비하고 있습니다...",
                              "results":None, "errors":[], "diagnostics":[], "market":market,
                              "universe_size":universe_size, "created_at":time.time()}
        st.session_state.task_id = task_id
        worker = threading.Thread(target=background_scan, args=(task_id, market, universe_size), daemon=True)
        worker.start()
        st.rerun()

# Refresh this page while a task is running to show progress; this does not itself run the scan.
try:
    from streamlit_autorefresh import st_autorefresh
    _refresh_available = True
except ImportError:
    _refresh_available = False

task_id = st.session_state.task_id
with TASKS_LOCK:
    task = TASKS.get(task_id, {}).copy() if task_id else None

if task and task.get("status") in ("queued", "running"):
    if _refresh_available:
        st_autorefresh(interval=2000, key="background_scan_refresh")
    st.info("백그라운드에서 조사 중이야. 이 화면을 닫지 않아도 되고 다른 앱으로 전환할 수도 있어.")
    st.progress(float(task.get("progress", 0.0)), text=task.get("progress_text", "조사 준비 중..."))
    st.caption(f"시장: {task.get('market')} · 검색 범위: {task.get('universe_size')}종목")
elif task and task.get("status") in ("done", "error"):
    if task_id != st.session_state.last_seen_finished_task:
        st.toast("주식 스캐너 조사가 끝났어.", icon="🔔")
        st.session_state.last_seen_finished_task = task_id
    if task.get("status") == "done":
        st.success("조사가 끝났어.")
        st.caption(task.get("notification", ""))
    else:
        st.error("조사 중 오류가 발생했어.")

st.session_state.scan_results = task.get("results") if task else None
st.session_state.scan_errors = task.get("errors", []) if task else []
st.session_state.scan_diagnostics = task.get("diagnostics", []) if task else []

results = st.session_state.scan_results
if results is None and (not task or task.get("status") not in ("queued", "running")):
    st.info("분석을 시작하려면 위의 **알려줘** 버튼을 한 번 누르세요.")
elif task and task.get("status") in ("queued", "running"):
    st.caption("조사가 진행 중이야. 완료되면 아래 결과가 자동으로 표시돼.")
elif not results:
    st.warning("분석 가능한 가격 데이터가 없습니다. 이는 진입 조건 때문이 아니라 데이터 수집/제공처 문제일 수 있습니다.")
    if st.session_state.scan_diagnostics:
        st.caption("분석 진단: " + " · ".join(st.session_state.scan_diagnostics))
    if st.session_state.scan_errors:
        with st.expander("오류 확인"):
            for err in st.session_state.scan_errors:
                st.write(f"- {err}")
    st.caption("조건을 충족하는 종목이 없을 때는 억지로 종목을 추천하지 않습니다.")
else:
    # Show exactly one actionable candidate. Other rows are not presented as recommendations.
    r = next((item for item in results if item["signal"]), None)
    if r is None:
        st.warning("모든 최종 진입 조건을 충족한 종목은 없습니다. 아래는 조건에 가까운 검색 후보이며, 매수 신호가 아닙니다.")
        st.caption("후보 검색 범위는 넓혔지만, 실제 진입 가능 판정 기준은 기존과 동일합니다.")
        near = sorted(
            results,
            key=lambda item: (
                item["fresh"], item["candidate_score"], item["trend"], item["breakout"],
                item["turnover_ratio"], item["volume_ratio"]
            ),
            reverse=True,
        )[:5]
        rows = []
        for item in near:
            rows.append({
                "종목": f'{item["name"]} ({item["code"]})',
                "기준일": item["date"],
                "데이터 상태": ("최신" if item["fresh"] else f"오래됨({item['age_days']}일)"),
                "종가(원)": round(item["close"]),
                "충족 조건": f'{item["candidate_score"]}/7',
                "거래대금(추정, 억 원)": round(item["turnover"] / 100_000_000, 2),
                "거래대금 증가": f'{item["turnover_ratio"]:.2f}배',
                "거래량 증가": f'{item["volume_ratio"]:.2f}배',
                "당일 등락률": f'{item["daily_pct"]:+.2f}%',
                "미충족 조건": ", ".join(item["failed_checks"]) or "없음",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.info("표의 종목은 확인용 후보입니다. 데이터 상태가 ‘오래됨’이면 진입 조건을 만족해 보여도 실제 진입 신호로 인정하지 않습니다.")
    else:
        st.success("진입 조건 충족 후보가 발견되었습니다. 아래 가격은 규칙 기반 참고값이며 주문 권고가 아닙니다.")
        st.caption(f"기준 데이터: {r['date']} · 데이터 나이: {r['age_days']}일 · 종목 코드: {r['code']}")
        st.subheader(f"{r['name']} ({r['code']})")
        st.metric("최근 종가", f"{r['close']:,.0f}원", f"{r['daily_pct']:+.2f}%")
        a, b, c = st.columns(3)
        a.metric("진입 기준가", f"{r['entry']:,.0f}원")
        b.metric("손절 기준가", f"{r['stop']:,.0f}원", f"-{(r['entry']-r['stop'])/r['entry']*100:.2f}%")
        c.metric("목표 기준가", f"{r['target']:,.0f}원", f"+{(r['target']-r['entry'])/r['entry']*100:.2f}%")
        st.subheader("30만 원 기준 매수 수량 계산")
        x, y, z = st.columns(3)
        x.metric("계산 수량", f"{r['shares']}주")
        y.metric("예상 사용 금액", f"{r['estimated_cost']:,.0f}원")
        z.metric("계획 손실 예산", f"{r['risk_budget']:,.0f}원")
        st.caption(
            f"계산상 손절 폭 기준 손실 예상액: {r['planned_risk']:,.0f}원(수수료·세금·슬리피지 제외). "
            "갭 하락 등으로 실제 손실은 더 커질 수 있습니다."
        )
        st.subheader("진입 조건 점검")
        checks = [
            ("종가 > 20일 이동평균 > 50일 이동평균", r["trend"]),
            ("종가가 직전 20거래일 고점을 돌파", r["breakout"]),
            ("추정 당일 거래대금 10억 원 이상", r["liquid"]),
            (f"거래대금이 직전 20일 평균의 {MIN_TURNOVER_RATIO:.1f}배 이상", r["turnover_ratio"] >= MIN_TURNOVER_RATIO),
            (f"거래량이 직전 20일 평균의 {MIN_VOLUME_RATIO:.1f}배 이상", r["volume_ratio"] >= MIN_VOLUME_RATIO),
            (f"당일 상승률 0% 이상 {MAX_DAILY_CHANGE_PCT:.1f}% 미만", 0 <= r["daily_pct"] < MAX_DAILY_CHANGE_PCT),
            (f"데이터가 {MAX_DATA_AGE_DAYS}일 이내", r["fresh"]),
        ]
        for label, ok in checks:
            st.write(("✅ " if ok else "❌ ") + label)
        st.subheader("보조 지표")
        d1, d2, d3 = st.columns(3)
        d1.metric("거래대금 증가", f"{r['turnover_ratio']:.2f}배")
        d2.metric("거래량 증가", f"{r['volume_ratio']:.2f}배")
        d3.metric("ATR(14)", f"{r['atr']:,.0f}원")
        export = pd.DataFrame([{
            "기준일": r["date"], "종목명": r["name"], "종목코드": r["code"],
            "최근종가(원)": round(r["close"]), "진입기준가(원)": round(r["entry"]),
            "손절기준가(원)": round(r["stop"]), "목표기준가(원)": round(r["target"]),
            "계산매수수량(주)": r["shares"], "예상사용금액(원)": round(r["estimated_cost"]),
            "계획손실예산(원)": round(r["risk_budget"]), "거래대금증가배수": round(r["turnover_ratio"], 2),
            "거래량증가배수": round(r["volume_ratio"], 2), "당일등락률(%)": round(r["daily_pct"], 2),
        }])
        st.download_button(
            "분석 결과 CSV 저장", export.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"스윙분석_{r['code']}_{r['date']}.csv", mime="text/csv", use_container_width=True,
        )
        if st.session_state.scan_errors:
            with st.expander("일부 종목 데이터 오류 확인"):
                st.write("일부 종목의 데이터 요청은 실패했지만, 유효한 종목 데이터는 계속 분석했습니다.")
                for err in st.session_state.scan_errors:
                    st.write(f"- {err}")

st.divider()
st.caption(
    "데이터는 FinanceDataReader를 통해 제공되는 일봉 데이터입니다. 추정 거래대금은 종가×거래량이며 "
    "공식 투자자별 순매수 자료가 아닙니다. 앱은 주문을 넣지 않습니다."
)
