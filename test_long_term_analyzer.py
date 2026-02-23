"""
중장기 투자 추천 시스템 테스트 및 사용 예제
"""
import pandas as pd
import sys
sys.path.insert(0, 'd:\\workspace\\StockVibe')

from long_term_analyzer import LongTermAnalyzer, create_investment_portfolio_recommendation

def test_long_term_analyzer():
    """테스트용 예제 데이터로 분석"""
    
    # 테스트 데이터 생성
    finance_df = pd.DataFrame({
        'code': ['005930', '000660', '035420', '003670', '005380'] * 3,
        'date': pd.date_range('2024-01-01', periods=3, freq='D').repeat(5),
        'roe': [12.5, 15.3, 18.2, 10.5, 14.2] * 3,
        'operating_margin': [9.2, 11.5, 13.1, 7.8, 10.3] * 3,
        'per': [14.5, 12.8, 16.2, 11.3, 15.7] * 3,
        'pbr': [0.95, 0.88, 1.05, 0.82, 0.98] * 3,
        'eps': [5000, 6000, 7000, 4000, 5500] * 3,
    })
    
    price_df = pd.DataFrame({
        'code': ['005930', '000660', '035420', '003670', '005380'] * 252,
        'date': pd.date_range('2023-01-01', periods=252, freq='D').repeat(5),
        'close': [70000, 80000, 120000, 50000, 100000] * 252,
    })
    
    # 아날라이저 생성
    analyzer = LongTermAnalyzer(finance_df, price_df)
    
    # 추천 종목 생성
    print("=" * 60)
    print("📊 중장기 투자 추천 시스템 테스트")
    print("=" * 60)
    
    recommendations = analyzer.recommend_long_term_stocks(
        num_stocks=5,
        min_fundamental_score=40,
        kospi_list={
            '005930': '삼성전자',
            '000660': 'SK하이닉스',
            '035420': 'NAVER',
            '003670': '포스코',
            '005380': 'CJ ENM'
        }
    )
    
    if not recommendations.empty:
        print("\n✅ 추천 종목:")
        print(recommendations[['code', 'name', 'total_score', 'fundamental_score', 
                              'valuation_score', 'momentum_score', 'roe', 'per']].to_string())
        
        # 첫 번째 종목 상세 분석
        first_code = recommendations.iloc[0]['code']
        first_name = recommendations.iloc[0]['name']
        
        print(f"\n{'=' * 60}")
        print(f"📊 {first_name} ({first_code}) 상세 분석")
        print(f"{'=' * 60}")
        
        details = analyzer.get_stock_recommendation_details(first_code, first_name)
        print(f"\n추천 레벨: {details['level']}")
        print(f"종합 점수: {details['total_score']:.1f}/100")
        print(f"\n📈 재무 지표:")
        print(f"  - ROE: {details['fundamental'].get('roe', 0):.1f}%")
        print(f"  - 영업이익률: {details['fundamental'].get('operating_margin', 0):.1f}%")
        print(f"\n💵 밸류에이션:")
        print(f"  - PER: {details['valuation'].get('per', 0):.1f}배")
        print(f"  - PBR: {details['valuation'].get('pbr', 0):.2f}배")
        print(f"\n🔮 투자 전망:")
        for outlook in details.get('outlook', []):
            print(f"  - {outlook}")
        
        # 포트폴리오 구성
        print(f"\n{'=' * 60}")
        print("💰 포트폴리오 구성")
        print(f"{'=' * 60}")
        
        portfolio_result = create_investment_portfolio_recommendation(
            recommendations.head(3),
            total_investment=10_000_000
        )
        
        print(f"\n전략: {portfolio_result['strategy']}")
        print(f"총 할당액: ₩{portfolio_result['total_allocation']:,.0f}")
        print(f"다변화 종목 수: {portfolio_result['diversification']}")
        print("\n📊 종목별 배분:")
        
        for portfolio in portfolio_result['portfolio']:
            allocation_ratio = (portfolio['allocation'] / portfolio_result['total_allocation'] * 100)
            print(f"  - {portfolio['name']} ({portfolio['code']}): "
                  f"₩{portfolio['allocation']:,.0f} ({allocation_ratio:.1f}%)")
    
    print("\n" + "=" * 60)
    print("✅ 테스트 완료!")
    print("=" * 60)

if __name__ == "__main__":
    test_long_term_analyzer()
