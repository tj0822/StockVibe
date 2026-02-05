"""GPU 감성분석 테스트"""
from sentiment_analyzer import SentimentAnalyzer

# 감성분석기 초기화 (GPU 사용)
analyzer = SentimentAnalyzer(use_model=True)

# 테스트 뉴스 텍스트
test_texts = [
    "삼성전자 주가가 급등하며 사상 최고가를 경신했습니다. 실적 개선 전망이 호재로 작용했습니다.",
    "반도체 업황 악화로 주요 기업들의 실적이 부진할 것으로 예상됩니다.",
    "내일 정기 주주총회가 예정되어 있습니다."
]

print("\n" + "="*70)
print("GPU 감성분석 테스트")
print("="*70)

for i, text in enumerate(test_texts, 1):
    result = analyzer.analyze(text)
    print(f"\n[테스트 {i}]")
    print(f"텍스트: {text}")
    print(f"결과: {result['emoji']} {analyzer.get_sentiment_text(result['sentiment'])}")
    print(f"점수: {result['score']}")
    print(f"방법: {result['method']}")
    if 'confidence' in result:
        print(f"신뢰도: {result['confidence']:.2%}")

print("\n" + "="*70)
