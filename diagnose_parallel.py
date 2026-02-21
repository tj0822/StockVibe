"""
병렬처리 동작 진단 스크립트
ProcessPoolExecutor vs ThreadPoolExecutor 확인
"""
import os
import sys
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

print("=" * 80)
print("병렬처리 환경 진단")
print("=" * 80)

# 1. 시스템 정보
print("\n✅ 시스템 정보:")
print(f"   - CPU 코어 수: {os.cpu_count()}")
print(f"   - Python 버전: {sys.version}")
print(f"   - OS: {sys.platform}")

# 2. ProcessPoolExecutor 테스트
print("\n" + "=" * 80)
print("ProcessPoolExecutor 테스트 (CPU-bound 작업)")
print("=" * 80)

def cpu_bound_task(n):
    """CPU 집약적 작업"""
    total = 0
    for i in range(n * 1000000):
        total += i
    return total

# 순차 처리
test_values = [10, 10, 10, 10]  # 4개 작업

print(f"\n📊 테스트: {len(test_values)}개 작업 (각각 CPU 계산 10M 반복)")

# 순차 처리 시간 측정
print("\n1️⃣ 순차 처리 (Sequential):")
start = time.time()
results_seq = [cpu_bound_task(v) for v in test_values]
seq_time = time.time() - start
print(f"   ⏱️  시간: {seq_time:.2f}초")
print(f"   ✅ 결과: {len(results_seq)}개 완료")

# ProcessPoolExecutor 테스트
print("\n2️⃣ ProcessPoolExecutor (병렬 처리):")
try:
    start = time.time()
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(cpu_bound_task, v) for v in test_values]
        results_proc = [f.result() for f in as_completed(futures)]
    proc_time = time.time() - start
    print(f"   ⏱️  시간: {proc_time:.2f}초")
    print(f"   ✅ 결과: {len(results_proc)}개 완료")
    print(f"   ⚡ 속도 향상: {seq_time / proc_time:.2f}배")
except Exception as e:
    print(f"   ❌ 오류: {e}")
    proc_time = None

# ThreadPoolExecutor 테스트
print("\n3️⃣ ThreadPoolExecutor (GIL 제한):")
try:
    start = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(cpu_bound_task, v) for v in test_values]
        results_thread = [f.result() for f in as_completed(futures)]
    thread_time = time.time() - start
    print(f"   ⏱️  시간: {thread_time:.2f}초")
    print(f"   ✅ 결과: {len(results_thread)}개 완료")
    print(f"   ⚠️  배속 (ThreadPool 대비): {seq_time / thread_time:.2f}배")
    print(f"   ⚠️  ProcessPool이 ThreadPool보다 {thread_time / proc_time:.2f}배 빠름" if proc_time else "")
except Exception as e:
    print(f"   ❌ 오류: {e}")

# 3. Multiprocessing 컨텍스트 확인
print("\n" + "=" * 80)
print("Multiprocessing 컨텍스트 (Windows 호환성)")
print("=" * 80)

print(f"\n현재 시작 방식 (multiprocessing.get_start_method()): {multiprocessing.get_start_method()}")
available_methods = multiprocessing.get_all_start_methods()
print(f"사용 가능한 방식: {available_methods}")

if sys.platform == 'win32':
    print("\n⚠️  Windows 감지!")
    print("   - Windows는 'spawn' 방식만 지원 (프로세스 재시작)")
    print("   - 전체 모듈이 pickleable해야 함")
    print("   - ProcessPoolExecutor 사용 시 주의 필요")

# 4. 현재 optimizer.py 상태 확인
print("\n" + "=" * 80)
print("optimizer.py 병렬처리 상태 확인")
print("=" * 80)

import sys
sys.path.insert(0, 'd:\\workspace\\StockVibe')

try:
    from optimizer import BacktestOptimizer, _test_combination_worker
    print("\n✅ optimizer 모듈 임포트 성공")
    
    # worker 함수 pickleable 확인
    import pickle
    try:
        pickled = pickle.dumps(_test_combination_worker)
        print("✅ _test_combination_worker 함수가 pickleable (좋음!)")
    except Exception as e:
        print(f"❌ _test_combination_worker 함수가 pickleable하지 않음: {e}")
        
except Exception as e:
    print(f"❌ optimizer 모듈 임포트 실패: {e}")

print("\n" + "=" * 80)
print("진단 완료")
print("=" * 80)
print("""
📌 결론:
- ProcessPoolExecutor가 작동하면: 1.5~4배 성능 향상 (CPU 코어 수에 따라)
- ThreadPoolExecutor로 fallback되면: GIL 때문에 성능 향상 거의 없음
- Windows에서는 'spawn' 방식 사용 (느릴 수 있음)

💡 개선 방법:
1. 로거를 추가해 어느 처리기를 사용하는지 확인
2. Windows에서도 ProcessPoolExecutor가 작동하도록 보장
3. ThreadPoolExecutor 사용 방지 (CPU-bound 작업에는 효과 없음)
""")
