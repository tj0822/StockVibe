#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import random
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


REQUEST_TIMEOUT = 10
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.7,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Referer": "https://finance.naver.com/",
        }
    )
    return session


def load_kospi_codes(data_dir: str) -> list[tuple[str, str]]:
    path = os.path.join(data_dir, "kospi_list.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일이 없습니다: {path}")

    raw = pd.read_pickle(path)
    pairs: list[tuple[str, str]] = []

    if isinstance(raw, dict):
        for code, name in raw.items():
            pairs.append((str(code).zfill(6), str(name)))
    elif isinstance(raw, pd.DataFrame):
        if "code" not in raw.columns:
            raise ValueError("kospi_list.pkl DataFrame에 code 컬럼이 없습니다.")
        name_col = "name" if "name" in raw.columns else None
        for _, row in raw.iterrows():
            code = str(row["code"]).zfill(6)
            name = str(row[name_col]) if name_col else ""
            pairs.append((code, name))
    else:
        raise ValueError("kospi_list.pkl 형식을 해석할 수 없습니다.")

    dedup = {}
    for code, name in pairs:
        dedup[code] = name
    return sorted(dedup.items(), key=lambda x: x[0])


def extract_sector_from_html(html_text: str) -> str | None:
    soup = BeautifulSoup(html_text, "lxml")

    # 1) 가장 안정적인 패턴: 업종 상세 링크 텍스트
    link = soup.select_one("a[href*='sise_group_detail.naver?type=upjong']")
    if link:
        link_text = " ".join(link.get_text(" ", strip=True).split())
        if link_text:
            return link_text

    for node in soup.find_all(string=re.compile(r"업종명\s*:")):
        text_line = " ".join(node.parent.get_text(" ", strip=True).split()) if node.parent else ""
        m = re.search(r"업종명\s*:\s*([^｜\)\]]+)", text_line)
        if m:
            sector = m.group(1).strip()
            if sector:
                return sector

    text_all = " ".join(soup.get_text(" ", strip=True).split())
    m = re.search(r"동종업종비교\s*\(업종명\s*:\s*([^｜\)\]]+)", text_all)
    if m:
        sector = m.group(1).strip()
        if sector:
            return sector

    m2 = re.search(r"업종명\s*:\s*([^｜\)\]]+)", text_all)
    if m2:
        sector = m2.group(1).strip()
        if sector:
            return sector

    return None


def fetch_sector(session: requests.Session, code: str, name: str) -> dict:
    result = {
        "code": code,
        "name": name,
        "sector": None,
        "source_url": f"https://finance.naver.com/item/main.naver?code={code}",
        "error": None,
    }

    try:
        resp = session.get(result["source_url"], timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        sector = None
        encodings = []
        if resp.encoding:
            encodings.append(resp.encoding)
        if resp.apparent_encoding:
            encodings.append(resp.apparent_encoding)
        encodings.extend(["euc-kr", "cp949", "utf-8"])

        unique_encodings = []
        for enc in encodings:
            if enc and enc not in unique_encodings:
                unique_encodings.append(enc)

        for enc in unique_encodings:
            try:
                html_text = resp.content.decode(enc, errors="replace")
                sector = extract_sector_from_html(html_text)
                if sector:
                    break
            except Exception:
                continue

        result["sector"] = sector
        if sector:
            logging.info("%s %s -> %s", code, name, sector)
        else:
            logging.warning("%s %s -> 업종명 미추출", code, name)
    except Exception as exc:
        result["error"] = str(exc)
        logging.error("%s %s -> 오류: %s", code, name, exc)

    return result


def save_results(results: list[dict], data_dir: str) -> tuple[str, str]:
    os.makedirs(data_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df = df.sort_values(["code"]).reset_index(drop=True)

    json_path = os.path.join(data_dir, "kospi_sector_info.json")
    pkl_path = os.path.join(data_dir, "kospi_sector_info.pkl")

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(df.to_dict(orient="records"), fh, ensure_ascii=False, indent=2)
    df.to_pickle(pkl_path)
    return json_path, pkl_path


def main() -> None:
    parser = argparse.ArgumentParser(description="네이버 금융 KOSPI 업종(섹터) 수집")
    parser.add_argument("--data-dir", default="data", help="데이터 디렉터리 경로")
    parser.add_argument("--sleep-min", type=float, default=0.3, help="요청 간 최소 대기(초)")
    parser.add_argument("--sleep-max", type=float, default=0.8, help="요청 간 최대 대기(초)")
    parser.add_argument("--limit", type=int, default=0, help="테스트용 종목 수 제한(0=전체)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    codes = load_kospi_codes(args.data_dir)
    if args.limit > 0:
        codes = codes[: args.limit]

    session = build_session()
    results: list[dict] = []

    total = len(codes)
    logging.info("수집 시작: %d개 종목", total)
    for idx, (code, name) in enumerate(codes, start=1):
        logging.info("[%d/%d] %s %s", idx, total, code, name)
        results.append(fetch_sector(session, code, name))
        if idx < total:
            time.sleep(random.uniform(args.sleep_min, args.sleep_max))

    json_path, pkl_path = save_results(results, args.data_dir)

    success_count = sum(1 for r in results if r.get("sector"))
    fail_count = total - success_count
    logging.info("완료: 총 %d개, 성공 %d개, 실패/미추출 %d개", total, success_count, fail_count)
    logging.info("저장(JSON): %s", json_path)
    logging.info("저장(PKL): %s", pkl_path)


if __name__ == "__main__":
    main()
