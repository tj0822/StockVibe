from bs4 import BeautifulSoup
import urllib.request
import ssl
import pandas as pd
import os
import datetime as dt
import streamlit as st
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

class CrawlingKospi:
    def __init__(self):
        pass

    def GetKospi200(self):     
        stockDic = dict()
        lastPageNum = 0

        # 마지막 페이지 찾기
        base_url = "https://finance.naver.com/sise/entryJongmok.nhn?&page="
        target_url = base_url + str(1)
        context = ssl._create_unverified_context()
        soup = BeautifulSoup(urllib.request.urlopen(target_url, context=context).read().decode('euc-kr', 'ignore'), "lxml")
        for item in soup.find_all('td'):
            if item.has_attr('class') and 'pgRR' in item['class']:
                lastPageNum = int(str(item.a['href']).replace('/sise/entryJongmok.nhn?&page=', ''))

        for i in range(1, lastPageNum+1):
            target_url = base_url + str(i)
            soup = BeautifulSoup(urllib.request.urlopen(target_url, context=context).read().decode('euc-kr', 'ignore'), "lxml")
            postNoList = soup.find_all('a')


            # 종목코드와 종목명 담기
            for item in postNoList:
                if item.has_attr('target') and '_parent' in item['target'] and item.has_attr('href'):
                    if str(item['href']).startswith('/item/main.naver?code='):                    
                        stockDic[str(item['href']).replace('/item/main.naver?code=', '')] = item.text
        st.write("KOSPI 리스트 수집완료!")
        return stockDic

    def get_all_kospi_data(self):
        """저장된 KOSPI 200 주가 데이터를 로드하여 반환"""
        stock_file = "data/stock.pkl"
        if os.path.exists(stock_file):
            df = pd.read_pickle(stock_file)
            # 데이터 타입 변환
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['open'] = pd.to_numeric(df['open'], errors='coerce')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            return df
        else:
            # 파일이 없으면 빈 DataFrame 반환
            return pd.DataFrame(columns=['date', 'code', 'open', 'close', 'low', 'high', 'volume'])

    def getKospiIndex(self):
        outFileName = "data/kospi_index.pkl"
        i = 1
        last_date = ""
        cols = ["date", "index", "diff", "diff_ratio", "amount", "tot_price"]
        if os.path.exists(outFileName):
            df = pd.read_pickle(outFileName)
        else:
            df = pd.DataFrame(columns=cols)
        while True:        
            url = "https://finance.naver.com/sise/sise_index_day.nhn?code=KOSPI&page=" + str(i)              
            table = pd.read_html(url, encoding='euc-kr')
            table = table[0].dropna() 
            table.columns = cols
            if last_date == table["date"].max() or dt.datetime.today().strftime('%H:%M') < "15:30":
                break
            else:
                last_date = table["date"].max()
                if len(table[~table["date"].isin(df["date"].tolist())]) == 0:
                    break
                
                df = pd.concat([df, table[~table["date"].isin(df["date"].tolist())]], axis=0)
                i += 1
        df.to_pickle(outFileName)
        # df.to_sql(
        #     name='kospi_index',
        #     con=self.engine,
        #     if_exists='append',
        #     index=False
        # )
        st.write("KOSPI Index 수집완료!")


    ''' 
    현재 실시간 가격 조회
    '''
    def GetCurrentPrice(self, code): 
        
        headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}
        url = "https://finance.naver.com/item/sise_day.nhn?code=" + code
        page = "&page="
        idx = 1

        stockPriceList = []
        
        fullAddr = url + page + str(idx)
        source_code = requests.get(fullAddr, headers = headers)
        source_code.encoding = 'euc-kr'
        if source_code is None:
            # print('source_code is None')            
            return

        soup = BeautifulSoup(source_code.text,"lxml")       
        
        tr = soup.find("tr", onmouseout=True)  # 첫 번째 tr만 선택
        if tr and tr.find("span", class_="tah p11") is not None:  # 가격 데이터가 있는 경우만 실행
            tDate = tr.find("span", class_="tah p10 gray03").text
            cPrice = tr.find_all("span", class_="tah p11")
            sIdx = 1

            if len(cPrice) != 5:
                sIdx = 2

            dt = tDate.replace(".", "-")

            pClose = float(cPrice[0].text.replace(",", ""))
            pOpen = float(cPrice[sIdx].text.replace(",", ""))
            sIdx += 1
            pHigh = float(cPrice[sIdx].text.replace(",", ""))
            sIdx += 1
            pLow = float(cPrice[sIdx].text.replace(",", ""))
            sIdx += 1
            volume = float(cPrice[sIdx].text.replace(",", ""))

            stockList = [
                dt, code, pOpen, pClose, pLow, pHigh, volume
            ]

            stockPriceList.append(stockList)

        return stockPriceList

    def GetPriceData(self, item): 
        
        # print(item)
        code = item[0]
        name = item[1]

        existing_dates = set(self.stockDf.loc[self.stockDf["code"] == code, "date"])

        headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}
        url = "https://finance.naver.com/item/sise_day.nhn?code=" + code
        page = "&page="
        idx = 1
        
            
        bFlag = True
        stockPriceList = []
        
        while bFlag == True:
            fullAddr = url + page + str(idx)
            # print(fullAddr)
            source_code = requests.get(fullAddr, headers = headers)
            source_code.encoding = 'euc-kr'
            if source_code is None:
                # print('source_code is None')
                bFlag = False
                return

            soup = BeautifulSoup(source_code.text,"lxml")
            if soup.find('td', class_='on').find('a').text != str(idx):
                # print('마지막 페이지')
                bFlag = False
                return
            
            for tr in soup.find_all("tr", onmouseout=True):
                if tr.find("span",class_ = "tah p11") is None:
                    # 가격데이터가 없으면 False로 빠져나옴
                    # print('no price data')
                    bFlag = False
                    break

                else:
                    tDate = tr.find("span",class_ = "tah p10 gray03").text
                    
                    
                    cPrice = tr.find_all("span",class_ = "tah p11")
                    sIdx = 1

                    if len(cPrice) != 5 :
                        sIdx = 2

                    date = tDate.replace("." ,"-")
                    if dt.datetime.today().strftime('%H:%M') < "15:30" and date == dt.datetime.today().strftime('%Y-%m-%d'):
                        continue

                    if date in existing_dates:                  
                        # print(dt, " 가격존재 ", item)
                        bFlag = False
                        continue

                    pClose = float(cPrice[0].text.replace("," ,""))
                    pStart = float(cPrice[sIdx].text.replace("," ,""))
                    sIdx += 1
                    pMax   = float(cPrice[sIdx].text.replace("," ,""))
                    sIdx += 1
                    pMin   = float(cPrice[sIdx].text.replace("," ,""))
                    sIdx += 1
                    amount = float(cPrice[sIdx].text.replace("," ,""))

                    stockList = []
                    # stockList.append([date, code, pStart, pClose, pMin, pMax, amount])
                    stockList.append(date)
                    stockList.append(code)
                    stockList.append(pStart)
                    stockList.append(pClose)
                    stockList.append(pMin)
                    stockList.append(pMax)
                    stockList.append(amount)    

                    stockPriceList.append(stockList)                                            
                    # stockPriceList.append([date, code, pStart, pClose, pMin, pMax, amount])
            idx += 1
        # print(pd.DataFrame(stockPriceList, columns=self.stockDf.columns))
        # return stockPriceList
        # print(stockPriceList)
        return pd.DataFrame(stockPriceList, columns=self.stockDf.columns)

    def getFinanceInfo(self, code=None):
        headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}
        # 투자정보
        # totalCnt = []  # 상장주식수
        # forignerHaveLimit = []  # 외국인한도보유주식수
        # forignerHaveCnt = []  # 외국인보유주식수
        # max52week = []  # 52주 최고
        # min52week = []  # 52주 최저
        # per = []
        # eps = []
        # per_eps_date = []
        # estimate_per = []  # 추정 PER
        # estimate_eps = []  # 추정 EPS
        # pbr = []
        # bps = []
        # pbr_bps_date = []
        # dvr = []  # 배당수익
        financeFileName = 'data/stock_finance_data.pkl'
        cols = ["date",
                "code", 
                "totalCnt", 
                "forignerHaveLimit",
                "forignerHaveCnt",
                "min52week",
                "max52week",
                "per",
                "eps",
                "per_eps_date",
                "estimate_per",
                "estimate_eps",
                "pbr",
                "bps",
                "pbr_bps_date",
                "dvr"]
        if os.path.exists(financeFileName):          
            df = pd.read_pickle(financeFileName)
        else:
            df = pd.DataFrame(columns=cols)
            
        url = "https://finance.naver.com/item/main.nhn?code=" + code

        source_code = requests.get(url, headers = headers)
        source_code.encoding = 'euc-kr'
        if source_code is None:
            pass
        else:
            soup = BeautifulSoup(source_code.text, "lxml")
            date = soup.find(id='time').find_all('em')[0].text            
            # 이미 존재하는 데이터 또는 장마감 전이면 skip
            if ((df["date"] == date) & (df["code"] == code)).sum() > 0 or dt.datetime.today().strftime('%H:%M') < "15:30":
                return
            
            totalCnt = soup.find(id='tab_con1').find_all('em')[2].text.replace(',', '').replace('N/A', '0')
            # print(totalCnt)
            forignerHaveLimit = soup.find(id='tab_con1').find_all('em')[5].text.replace(',', '').replace('N/A', '0')
            forignerHaveCnt = soup.find(id='tab_con1').find_all('em')[6].text.replace(',', '').replace('N/A', '0')
            max52week = soup.find(id='tab_con1').find_all('em')[10].text.replace(',', '').replace('N/A', '0')
            min52week = soup.find(id='tab_con1').find_all('em')[11].text.replace(',', '').replace('N/A', '0')
            per = soup.find(id='tab_con1').find_all('em')[12].text.replace(',', '').replace('N/A', '0')
            eps = soup.find(id='tab_con1').find_all('em')[13].text.replace(',', '').replace('N/A', '0')
            per_eps_date = soup.find(id='tab_con1').find_all('span', class_='date')[0].text.replace('(','').replace(')','') if len(soup.find(id='tab_con1').find_all('span', class_='date')) > 0 else ""
            estimate_per = soup.find(id='tab_con1').find_all('em')[14].text.replace(',', '').replace('N/A', '0')
            estimate_eps = soup.find(id='tab_con1').find_all('em')[15].text.replace(',', '').replace('N/A', '0')
            pbr = soup.find(id='tab_con1').find_all('em')[16].text.replace(',', '').replace('N/A', '0')
            bps = soup.find(id='tab_con1').find_all('em')[17].text.replace(',', '').replace('N/A', '0')
            pbr_bps_date = soup.find(id='tab_con1').find_all('span', class_='date')[1].text.replace('(','').replace(')','') if len(soup.find(id='tab_con1').find_all('span', class_='date')) > 1 else ""
            dvr = soup.find(id='tab_con1').find_all('em')[18].text.replace(',', '').replace('N/A', '0')                
            
            financeList = []
            financeList.append(date)
            financeList.append(str(code).zfill(6))
            financeList.append(totalCnt)
            financeList.append(forignerHaveLimit)
            financeList.append(forignerHaveCnt)        
            financeList.append(max52week)
            financeList.append(min52week)
            financeList.append(per)
            financeList.append(eps)
            financeList.append(per_eps_date)
            financeList.append(estimate_per)
            financeList.append(estimate_eps)
            financeList.append(pbr)
            financeList.append(bps)
            financeList.append(pbr_bps_date)
            financeList.append(dvr)    
            
            # print(financeList)
            return pd.DataFrame([financeList], columns=cols)


    def crawling(self):
        # from sqlalchemy import create_engine
        # self.engine = create_engine(
        #     'mysql+pymysql://tj0822:tj0822@172.28.213.28:3306/mydb?charset=utf8mb4',
        #     pool_pre_ping=True,
        #     pool_recycle=3600
        # )

        directory = 'data/'
        kospiListFile = "kospi_list.pkl"
        stockDict = self.GetKospi200()

        if os.path.exists(directory + 'stock.pkl'):        
            self.stockDf = pd.read_pickle(directory + 'stock.pkl')
            # self.stockDf = pd.read_sql_table('stock', con=self.engine)
            # if dt.datetime.today().strftime('%H:%M') < "15:30":
            #     self.stockDf = self.stockDf[~(self.stockDf["date"] == dt.datetime.today().strftime('%Y-%m-%d'))]
            

        bFlag = False
        
        if os.path.exists(directory+kospiListFile):
            kospiListDf = pd.read_pickle(directory+kospiListFile)
            # kospiListDf = pd.read_sql_table('kospi_list', con=self.engine)
            # bFlag = bool((kospiListDf["date"] == dt.datetime.today().strftime("%Y-%m-%d")).sum())
            kospiListDf["date"] = pd.to_datetime(kospiListDf["date"]).dt.strftime("%Y-%m-%d")
            bFlag = (kospiListDf["date"] == dt.datetime.today().strftime("%Y-%m-%d")).any()

        
        if os.path.exists(directory+'stock.pkl'):
            crawlingPriceDf = pd.read_pickle(directory+'stock.pkl')
            # crawlingPriceDf = pd.read_sql_table('stock', con=self.engine)
        else:
            crawlingPriceDf = pd.DataFrame()
            # crawlingPriceDf = pd.read_sql_table('stock', con=self.engine)
        if os.path.exists(directory+'stock_finance_data.pkl'):            
            crawlingFinanceDf = pd.read_pickle(directory+'stock_finance_data.pkl')
            # crawlingFinanceDf = pd.read_sql_table('stock_finance', con=self.engine)
        else:
            crawlingFinanceDf = pd.DataFrame()
            # crawlingFinanceDf = pd.read_sql_table('stock_finance', con=self.engine)

        kospiList = []
        
        # 병렬 처리로 크롤링 속도 개선
        total_stocks = len(stockDict)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 데이터 저장용 리스트
        price_dfs = []
        finance_dfs = []
        
        # 병렬 처리 함수
        def fetch_stock_data(code_name_pair):
            code, name = code_name_pair
            try:
                price_df = self.GetPriceData((code, name))
                finance_df = self.getFinanceInfo(code)
                return code, name, price_df, finance_df, None
            except Exception as e:
                return code, name, None, None, str(e)
        
        # ThreadPoolExecutor로 병렬 처리 (최대 10개 동시 실행)
        max_workers = min(10, total_stocks)
        completed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 모든 작업 제출
            future_to_stock = {
                executor.submit(fetch_stock_data, (code, stockDict[code])): code 
                for code in stockDict.keys()
            }
            
            # 완료된 작업 처리
            for future in as_completed(future_to_stock):
                code, name, price_df, finance_df, error = future.result()
                
                if error:
                    st.warning(f"⚠️ {name}({code}) 데이터 수집 실패: {error[:50]}")
                else:
                    if not bFlag:
                        kospiList.append([dt.datetime.today().strftime("%Y-%m-%d"), code, name])
                    
                    if price_df is not None and not price_df.empty:
                        price_dfs.append(price_df)
                    if finance_df is not None and not finance_df.empty:
                        finance_dfs.append(finance_df)
                
                # 진행률 업데이트
                completed += 1
                progress = completed / total_stocks
                progress_bar.progress(progress)
                status_text.text(f"크롤링 진행 중... {completed}/{total_stocks} ({progress*100:.1f}%)")
        
        # 데이터 병합
        if price_dfs:
            crawlingPriceDf = pd.concat([crawlingPriceDf.dropna(axis=1, how='all')] + price_dfs, axis=0)
        if finance_dfs:
            crawlingFinanceDf = pd.concat([crawlingFinanceDf] + finance_dfs, axis=0)
        
        progress_bar.empty()
        status_text.empty()
        
        # crawlingPriceDf = pd.concat([crawlingPriceDf.dropna(axis=1, how='all'), pd.DataFrame(cralingPrice, columns=crawlingPriceDf.columns)], axis=0)

        if kospiList:                
            pd.concat([kospiListDf, pd.DataFrame(kospiList, columns=['date', 'code', 'name'])], axis=0).to_pickle(directory+kospiListFile)
            # new_list_df = pd.DataFrame(kospiList, columns=['date', 'code', 'name'])
            # new_list_df.to_sql(
            #     name='kospi_list',
            #     con=self.engine,
            #     if_exists='append',
            #     index=False
            # )

        crawlingPriceDf.to_pickle('data/stock.pkl')
        # crawled = crawlingPriceDf.copy()
        # crawled['date'] = pd.to_datetime(crawled['date'], errors='coerce')
        # for col in ['open', 'close', 'low', 'high', 'volume']:
        #     crawled[col] = pd.to_numeric(crawled[col], errors='coerce')
        # crawled.to_sql(
        #     name='stock',
        #     con=self.engine,
        #     if_exists='append',
        #     index=False,
        #     chunksize=1000
        # )
        st.write("✔ price Data 수집완료!")

        crawlingFinanceDf.to_pickle('data/stock_finance_data.pkl')
        fin = crawlingFinanceDf.copy()
        # fin['date'] = pd.to_datetime(fin['date'], errors='coerce')
        # 필요시 추가적인 컬럼 타입 변환
        # fin.to_sql(
        #     name='stock_finance',
        #     con=self.engine,
        #     if_exists='append',
        #     index=False
        # )
        st.write("✔ Finance Data 수집완료!")
        self.getKospiIndex()