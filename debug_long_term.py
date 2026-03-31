"""
중장기 투자 분석기 데이터 형식 확인 및 디버깅
"""
import sys
sys.path.insert(0, 'd:\\workspace\\StockVibe')

from app.data import load_finance_data, load_stock_data
from long_term_analyzer import LongTermAnalyzer

def check_data_format():
    """로드된 데이터 형식 확인"""
    
    print("=" * 70)
    print("📊 데이터 형식 확인")
    print("=" * 70)
    
    try:
        # 재무 데이터 로드
        print("\n📋 재무 데이터 로드 중...")
        finance_df = load_finance_data("data")
        
        if finance_df.empty:
            print("❌ 재무 데이터가 비어있습니다!")
            print("메인 대시보드에서 데이터를 먼저 수집하세요.")
        else:
            print(f"✅ 재무 데이터: {len(finance_df)}행")
            print(f"\n컬럼명:")
            print(finance_df.columns.tolist())
            
            print(f"\n샘플 데이터:")
            print(finance_df.head(3))
            
            print(f"\n데이터 타입:")
            print(finance_df.dtypes)
            
            # 필수 컬럼 확인
            print(f"\n필수 컬럼 확인:")
            required_cols = ['code', 'date', 'roe', 'per', 'pbr', 'eps']
            for col in required_cols:
                if col in finance_df.columns:
                    print(f"  ✅ {col}: 있음")
                else:
                    print(f"  ❌ {col}: 없음")
        
        # 주가 데이터 로드
        print("\n" + "=" * 70)
        print("\n📈 주가 데이터 로드 중...")
        price_df = load_stock_data("data")
        
        if price_df.empty:
            print("❌ 주가 데이터가 비어있습니다!")
        else:
            print(f"✅ 주가 데이터: {len(price_df)}행")
            print(f"\n컬럼명:")
            print(price_df.columns.tolist())
            
            print(f"\n샘플 데이터:")
            print(price_df.head(3))
            
            # 필수 컬럼 확인
            print(f"\n필수 컬럼 확인:")
            required_cols = ['code', 'date', 'close']
            for col in required_cols:
                if col in price_df.columns:
                    print(f"  ✅ {col}: 있음")
                else:
                    print(f"  ❌ {col}: 없음")
        
        # 분석 시도
        if not finance_df.empty and not price_df.empty:
            print("\n" + "=" * 70)
            print("\n🔍 분석 시도 (최소 점수: 30점)...")
            
            analyzer = LongTermAnalyzer(finance_df, price_df)
            recommendations = analyzer.recommend_long_term_stocks(
                num_stocks=10,
                min_fundamental_score=30
            )
            
            if recommendations.empty:
                print("⚠️ 추천 종목 없음")
                
                # 각 종목별 점수 확인
                print("\n🔍 개별 종목 분석 (최대 5개):")
                codes = finance_df['code'].unique()[:5]
                for code in codes:
                    fundamental = analyzer.analyze_fundamentals(code)
                    print(f"  {code}: 재무점수={fundamental.get('score', 0)}")
            else:
                print(f"\n✅ {len(recommendations)}개 종목 발굴!")
                print("\n추천 종목:")
                print(recommendations[['name', 'code', 'total_score', 'fundamental_score']].to_string())
    
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    check_data_format()
