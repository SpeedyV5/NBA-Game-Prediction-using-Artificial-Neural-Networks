from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import requests
from bs4 import BeautifulSoup

# ================== BOOT ==================
print("[BOOT] download_injury_report.py başladı")
print("[BOOT] python:", sys.executable)
print("[BOOT] cwd:", Path.cwd())
# ==========================================

NBA_INJURY_PAGE_DEFAULT = "https://official.nba.com/nba-injury-report-2025-26-season/"
NBA_STATIC_PREFIX = "https://ak-static.cms.nba.com"


# ================== DOWNLOAD ==================
def _http_get(url: str, *, timeout: int = 30) -> requests.Response:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp


def download_latest_injury_pdf(
    base_url: str = NBA_INJURY_PAGE_DEFAULT,
    save_dir: Path = Path("data_raw/injury_reports_raw"),
) -> Optional[Path]:
    print("[INFO] Injury page:", base_url)

    try:
        resp = _http_get(base_url)
    except Exception as e:
        print(f"❌ Sayfa alınamadı: {e}")
        return None

    soup = BeautifulSoup(resp.content, "html.parser")
    links = soup.select("a[href*='Injury-Report_']")

    pdf_links: List[Tuple[str, str]] = []
    for a in links:
        href = (a.get("href") or "").strip()
        if href.endswith(".pdf") and "Injury-Report_" in href:
            full = href if href.startswith("http") else f"{NBA_STATIC_PREFIX}{href}"
            ts = full.split("Injury-Report_")[1].replace(".pdf", "")
            pdf_links.append((ts, full))

    if not pdf_links:
        print("❌ PDF bulunamadı")
        return None

    def parse_ts(ts: str):
        for f in ("%Y-%m-%d_%I%p", "%Y-%m-%d_%I%M%p"):
            try:
                return datetime.strptime(ts, f)
            except Exception:
                pass
        return datetime.min

    pdf_links.sort(key=lambda x: parse_ts(x[0]), reverse=True)
    ts, url = pdf_links[0]

    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"Injury-Report_{ts}.pdf"

    print("[INFO] PDF indiriliyor:", url)
    out.write_bytes(_http_get(url).content)
    print("[OK] PDF indirildi:", out)

    return out


# ================== PARSER ==================
DATE_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")
TIME_RE = re.compile(r"^\d{1,2}:\d{2}$")
MATCHUP_RE = re.compile(r"^[A-Z]{2,3}@[A-Z]{2,3}$")
STATUS_RE = re.compile(r"\b(Out|Questionable|Probable|Available|Doubtful)\b", re.I)
REPORT_TS_RE = re.compile(r"^Injury Report:\s*(.*)$", re.I)

STATUS_WORDS = r"(Out|Questionable|Probable|Available|Doubtful)"
PLAYER_START_RE = re.compile(rf"([A-Za-z\.\'-]+,\s*[A-Za-z\.\'-]+)\s+{STATUS_WORDS}\b")


def _clean(s: str) -> str:
    s = (s or "").strip()
    # pdfplumber bazen boşlukları yutuyor
    s = s.replace("NOTYETSUBMITTED", "NOT YET SUBMITTED")
    s = s.replace("(ET)", " (ET)")
    s = re.sub(r"(\d{1,2}:\d{2})\(", r"\1 (", s)
    s = re.sub(r"\s+", " ", s)
    return s


def split_multi_player_line(line: str) -> List[str]:
    """
    Bazen aynı satıra birden fazla oyuncu yapışıyor.
    Player+Status başlangıçlarına göre parçala.
    """
    matches = list(PLAYER_START_RE.finditer(line))
    if len(matches) <= 1:
        return [line]
    parts: List[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(line)
        parts.append(line[start:end].strip())
    return parts


def parse_injury_pdf_to_csv(pdf_path: Path) -> Optional[Path]:
    print("[INFO] Parsing:", pdf_path)

    try:
        import pdfplumber
        import pandas as pd
    except ModuleNotFoundError:
        print("❌ pdfplumber yok → pip install pdfplumber pandas")
        return None

    rows: List[Dict[str, Any]] = []

    report_dt: Optional[str] = None
    cur_date: Optional[str] = None
    cur_time: Optional[str] = None
    cur_matchup: Optional[str] = None
    cur_team: Optional[str] = None

    pending_reason: Optional[int] = None  # reason taşması için hedef satır

    def add(player: Optional[str], status: str, reason: str, raw: str):
        nonlocal pending_reason
        rows.append(
            dict(
                report_datetime=report_dt,
                game_date=cur_date,
                game_time_et=cur_time,
                matchup=cur_matchup,
                team=cur_team,
                player_name=player,
                status=status,
                reason=reason,
                raw_line=raw,
            )
        )
        idx = len(rows) - 1

        # Pending sadece oyuncu satırı olup reason boşsa açılır; yoksa kapanır
        if player and status != "NOT_YET_SUBMITTED" and (reason or "").strip() == "":
            pending_reason = idx
        else:
            pending_reason = None

    def append_reason(text: str):
        nonlocal pending_reason
        if pending_reason is None:
            return
        text = _clean(text)
        if not text:
            return
        prev = rows[pending_reason].get("reason") or ""
        rows[pending_reason]["reason"] = _clean((prev + " " + text).strip())

        # ';' ile bitmiyorsa genelde reason tamamlanır (örn. "Contusion")
        if not text.endswith(";"):
            pending_reason = None

    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for raw in (page.extract_text() or "").split("\n"):
                raw = _clean(raw)
                if not raw:
                    continue

                m = REPORT_TS_RE.match(raw)
                if m:
                    report_dt = _clean(m.group(1))
                    continue

                if raw.startswith("Page ") or raw.startswith("Game Date"):
                    continue

                for line in split_multi_player_line(raw):
                    line = _clean(line)
                    if not line:
                        continue

                    # 1) Eğer pending açıksa ve bu satırda status yoksa -> continuation
                    # (PDF’te Smith/Herro gibi örneklerde reason alt satıra taşıyor) :contentReference[oaicite:1]{index=1}
                    if pending_reason is not None and STATUS_RE.search(line) is None:
                        append_reason(line)
                        continue

                    # 2) NOT YET SUBMITTED satırları (date/time/matchup/team parse)
                    if "NOT YET SUBMITTED" in line:
                        toks = line.replace("NOT YET SUBMITTED", "").split()
                        i = 0

                        if i < len(toks) and DATE_RE.match(toks[i]):
                            cur_date = toks[i]; i += 1

                        if i < len(toks) and TIME_RE.match(toks[i]):
                            cur_time = toks[i]; i += 1
                            if i < len(toks) and toks[i] == "(ET)":
                                i += 1

                        if i < len(toks) and MATCHUP_RE.match(toks[i]):
                            cur_matchup = toks[i].upper(); i += 1

                        team_tokens = toks[i:]
                        cur_team = " ".join(team_tokens) if team_tokens else cur_team

                        add(None, "NOT_YET_SUBMITTED", "", line)
                        continue

                    # 3) Player row
                    ms = STATUS_RE.search(line)
                    if ms:
                        status = ms.group(1).capitalize()
                        left = _clean(line[: ms.start()])
                        right = _clean(line[ms.end():]).strip(" -;:")

                        toks = left.split()
                        i = 0
                        if i < len(toks) and DATE_RE.match(toks[i]):
                            cur_date = toks[i]; i += 1
                        if i < len(toks) and TIME_RE.match(toks[i]):
                            cur_time = toks[i]; i += 1
                            if i < len(toks) and toks[i] == "(ET)":
                                i += 1
                        if i < len(toks) and MATCHUP_RE.match(toks[i]):
                            cur_matchup = toks[i].upper(); i += 1

                        rem = toks[i:]
                        comma = next((k for k, t in enumerate(rem) if "," in t), None)

                        team = rem[:comma] if comma is not None else rem[:-2]
                        player = rem[comma:] if comma is not None else rem[-2:]

                        if team:
                            cur_team = " ".join(team)

                        add(" ".join(player), status, right, line)
                        continue

                    # 4) Eğer buraya geldiyse: status yok, pending yok -> ignore
                    # (bu satırlar genelde header/boşluk/garip kırılımlar)
                    continue

    import pandas as pd
    df = pd.DataFrame(rows)
    out = pdf_path.parent / "latest_injury.csv"
    df.to_csv(out, index=False)

    print("[OK] CSV:", out)
    if len(df):
        print("Team missing rate:", df["team"].isna().mean())
    return out


# ================== MAIN ==================
def main():
    pdf = download_latest_injury_pdf()
    if not pdf:
        sys.exit(1)

    out_csv = parse_injury_pdf_to_csv(pdf)
    if out_csv is None:
        sys.exit(2)

    print("[DONE] Injury pipeline tamamlandı")


if __name__ == "__main__":
    main()
