import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests 
import re

class NaverNewsCrawler:
    def __init__(self):
        pass

    def get_recent_news(self, stock_code, max_news=20):
        """
        네이버 금융에서 특정 종목의 최근 1주일간 뉴스 제목, 정보 제공 출처, 날짜, 뉴스 링크를 크롤링하는 함수
        """
        main_url = f"https://finance.naver.com/item/news.naver?code={stock_code}"
        headers = {"User-Agent": "Mozilla/5.0", "Referer": main_url}
        response = requests.get(main_url, headers=headers)
        
        if response.status_code != 200:
            print("[오류] 네이버 금융 메인 페이지에 접근할 수 없습니다.")
            return pd.DataFrame()
        
        soup = BeautifulSoup(response.text, "html.parser")
        iframe = soup.find("iframe", id="news_frame")
        
        if not iframe or "src" not in iframe.attrs:
            print("[오류] 뉴스 iframe을 찾을 수 없습니다.")
            return pd.DataFrame()
        
        # iframe에서 실제 뉴스 URL 가져오기 (page=를 빈 값으로 유지)
        base_news_url = "https://finance.naver.com" + iframe["src"]
        news_url = base_news_url.replace("page=1", "page=")  # 기본 페이지 처리
        
        news_response = requests.get(news_url, headers=headers)
        
        if news_response.status_code != 200:
            print(f"[오류] 네이버 뉴스 페이지에 접근할 수 없습니다. (URL: {news_url})")
            return pd.DataFrame()
        
        news_soup = BeautifulSoup(news_response.text, "html.parser")
        news_table = news_soup.select("table.type5 tbody tr")
        
        if not news_table:
            print("[오류] 뉴스 데이터를 찾을 수 없습니다. (iframe URL을 확인하세요)")
            return pd.DataFrame()
        
        news_list = []
        today = datetime.today()
        one_week_ago = today - timedelta(days=7)
        
        for row in news_table:
            cols = row.find_all("td")
            if len(cols) < 3:
                continue
            
            title_tag = cols[0].find("a")
            if not title_tag:
                continue
            
            title = title_tag.get_text(strip=True)
            news_link = title_tag["href"] if title_tag.has_attr("href") else ""
           
            source = cols[1].get_text(strip=True)
            date_str = cols[2].get_text(strip=True)
            
            try:
                news_date = datetime.strptime(date_str.strip(), "%Y.%m.%d %H:%M")
            except ValueError:
                continue
            
            if news_date >= one_week_ago:
                if news_link.startswith("/"):
                    full_news_link = "https://finance.naver.com" + news_link
                else:
                    full_news_link = news_link
                
                news_list.append({
                    "제목": title,
                    "출처": source,
                    "날짜": news_date.strftime("%Y-%m-%d %H:%M"),
                    "링크": full_news_link
                })
            
            if len(news_list) >= max_news:
                break
        
        return pd.DataFrame(news_list)

    def get_news_body(self, url):
        headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}
        source_code = requests.get(url, headers = headers)
        html_text = BeautifulSoup(source_code.text, "html.parser")
        # script 태그 찾기
        script_tag = html_text.find('script')
        if script_tag and script_tag.string:
            # script 태그 내부의 텍스트 추출
            script_content = script_tag.string
            # 정규표현식으로 URL 추출
            pattern = r"location\.href\s*=\s*['\"]([^'\"]+)['\"]"
            match = re.search(pattern, script_content)
            if match:
                url = match.group(1)
                # print("추출된 URL:", url)
            else:
                print("URL을 찾을 수 없습니다.")
        else:
            print("script 태그를 찾을 수 없습니다.")
            
        source_code = requests.get(url, headers = headers)
        html_text = BeautifulSoup(source_code.text, "html.parser")
        # print(html_text.find(id="articleTitle"))
        # print(html_text.find(id="articleBodyContents").text)
        # print(re.sub('<.+?>', '', html_text.find(id="dic_area").text, 0).strip())
        if html_text.find(id="dic_area") is None:
            print(url)
            return ""
        return re.sub('<.+?>', '', html_text.find(id="dic_area").text, 0).strip()
    
