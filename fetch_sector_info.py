#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
네이버 금융에서 KOSPI 200 종목의 정확한 섹터 정보를 가져오기
"""

import requests
from bs4 import BeautifulSoup
import time
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 테스트할 일부 종목 코드
test_codes = {
    '005930': '삼성전자',
    '000660': 'SK하이닉스',
    '005380': '현대자동차',
    '012330': '기아',
    '005490': '현대모비스',
    '051910': 'LG화학',
}

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

REQUEST_TIMEOUT = 5
REQUEST_SLEEP_SECONDS = 0.5


def build_session():
    """재시도 정책이 적용된 requests Session 생성"""
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
    return session


def extract_sector_from_soup(soup):
    """HTML에서 업종 텍스트를 우선순위 기반으로 추출"""
    label_candidates = soup.select('th, dt, strong, span')
    for label in label_candidates:
        label_text = label.get_text(strip=True)
        if '업종' not in label_text:
            continue

        next_cell = label.find_next_sibling(['td', 'dd'])
        if next_cell:
            value = next_cell.get_text(strip=True)
            if value:
                return value

        next_link = label.find_next('a')
        if next_link:
            value = next_link.get_text(strip=True)
            if value and '업종' not in value:
                return value

    # fallback: 테이블 row 단위 탐색
    for row in soup.select('tr'):
        headers_in_row = row.find_all(['th', 'dt'])
        values_in_row = row.find_all(['td', 'dd'])
        for header in headers_in_row:
            if '업종' in header.get_text(strip=True) and values_in_row:
                value = values_in_row[0].get_text(strip=True)
                if value:
                    return value

    return None

def get_sector_from_naver(session, code, name):
    """네이버 금융에서 종목의 섹터 정보 추출"""
    result = {
        'code': code,
        'name': name,
        'sector': None,
        'error': None,
    }

    try:
        url = f'https://finance.naver.com/item/main.naver?code={code}'
        response = session.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        sector = extract_sector_from_soup(soup)
        result['sector'] = sector
        logging.info('%s: %s → %s', code, name, sector if sector else '정보없음')
        return result
        
    except requests.RequestException as error:
        result['error'] = f'HTTP 오류: {error}'
        logging.error('%s: %s → %s', code, name, result['error'])
        return result
    except Exception as error:
        result['error'] = str(error)
        logging.error('%s: %s → 파싱 오류: %s', code, name, error)
        return result

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    print("=" * 50)
    print("네이버 금융 섹터 정보 크롤링")
    print("=" * 50)

    session = build_session()
    results = []

    for code, name in test_codes.items():
        result = get_sector_from_naver(session, code, name)
        results.append(result)
        time.sleep(REQUEST_SLEEP_SECONDS)  # 서버 부하 방지

    success_count = sum(1 for item in results if item['sector'])
    error_count = sum(1 for item in results if item['error'])
    logging.info('요약: 총 %d건, 섹터 추출 %d건, 오류 %d건', len(results), success_count, error_count)

    print("\n결론: 네이버 금융에서 섹터 정보 수집 가능 여부 확인됨")
