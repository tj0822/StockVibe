#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
네이버 금융에서 KOSPI 200 종목의 정확한 섹터 정보를 가져오기
"""

import requests
from bs4 import BeautifulSoup
import time

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

def get_sector_from_naver(code, name):
    """네이버 금융에서 종목의 섹터 정보 추출"""
    try:
        url = f'https://finance.naver.com/item/main.naver?code={code}'
        response = requests.get(url, headers=headers, timeout=5)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 종목 정보 섹션 찾기
        sector = None
        
        # 방법 1: em 태그에서 업종 찾기
        all_em = soup.find_all('em')
        for em in all_em:
            text = em.get_text(strip=True)
            if text and not text.isdigit():
                # 업종이 될 수 있는 텍스트 (숫자가 아닌 것)
                parent = em.find_parent()
                if parent:
                    parent_text = parent.get_text(strip=True)
                    if '업종' in parent_text or len(text) > 2:
                        sector = text
                        break
        
        # 방법 2: 테이블에서 찾기
        if not sector:
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for i, row in enumerate(rows):
                    cells = row.find_all(['td', 'th'])
                    for j, cell in enumerate(cells):
                        if '업종' in cell.get_text():
                            # 다음 셀이 값일 가능성
                            if j + 1 < len(cells):
                                sector = cells[j + 1].get_text(strip=True)
                            break
                if sector:
                    break
        
        print(f"{code}: {name} → {sector if sector else '정보없음'}")
        return sector
        
    except Exception as e:
        print(f"{code}: {name} → 오류: {e}")
        return None

if __name__ == '__main__':
    print("=" * 50)
    print("네이버 금융 섹터 정보 크롤링")
    print("=" * 50)
    
    for code, name in test_codes.items():
        sector = get_sector_from_naver(code, name)
        time.sleep(0.5)  # 서버 부하 방지
    
    print("\n결론: 네이버 금융에서 섹터 정보 수집 가능 여부 확인됨")
