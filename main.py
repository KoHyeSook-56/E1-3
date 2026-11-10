"""Mini NPU simulator. Python 3.8+, standard library only."""
import argparse
import json
import math
from pathlib import Path
import re
import time

EPSILON = 1e-9


def normalize_label(value):
    if not isinstance(value, str):
        raise ValueError('라벨은 문자열이어야 합니다.')
    labels = {'+': 'Cross', 'cross': 'Cross', 'x': 'X'}
    try:
        return labels[value.strip().lower()]
    except KeyError:
        raise ValueError('알 수 없는 라벨: {!r}'.format(value)) from None


def validate_matrix(matrix, n):
    if not isinstance(matrix, list) or len(matrix) != n:
        raise ValueError('{}개 행이 필요합니다.'.format(n))
    for row in matrix:
        if not isinstance(row, list) or len(row) != n:
            raise ValueError('각 행에 {}개 열이 필요합니다.'.format(n))
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError('배열 원소는 숫자여야 합니다.')
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError('유한한 숫자만 사용할 수 있습니다.')
    return matrix


def mac(pattern, kernel):
    """Validated square matrices: N² multiply-accumulate steps."""
    score = 0.0
    for i in range(len(pattern)):
        for j in range(len(pattern)):
            score += pattern[i][j] * kernel[i][j]
    if not math.isfinite(score):
        raise ValueError('MAC 결과가 유한하지 않습니다(수치 범위 초과).')
    return score


def flatten_matrix(matrix):
    """Convert a validated matrix to row-major storage."""
    return [value for row in matrix for value in row]


def mac_flat(pattern, kernel):
    """Multiply-accumulate equally sized, row-major arrays."""
    if len(pattern) != len(kernel):
        raise ValueError('1차원 배열의 크기가 서로 다릅니다.')
    score = 0.0
    for index in range(len(pattern)):
        score += pattern[index] * kernel[index]
    if not math.isfinite(score):
        raise ValueError('MAC 결과가 유한하지 않습니다(수치 범위 초과).')
    return score


def decide(a, b, labels=('Cross', 'X')):
    if abs(a - b) < EPSILON:
        return 'UNDECIDED'
    return labels[0] if a > b else labels[1]


def generate_patterns(n):
    if n < 3 or n % 2 == 0:
        raise ValueError('패턴 크기는 3 이상의 홀수여야 합니다.')
    cross = [[int(i == n // 2 or j == n // 2) for j in range(n)] for i in range(n)]
    x = [[int(i == j or i + j == n - 1) for j in range(n)] for i in range(n)]
    return cross, x


def benchmark(sizes, repeats=100, samples=None):
    if repeats < 10:
        raise ValueError('반복 횟수는 10 이상이어야 합니다.')
    print('\n[성능 분석: MAC 1회 평균, 동일 입력·반복 {}회]'.format(repeats))
    print('평탄화 준비 시간은 제외 (배열 변환은 측정 전에 1회 수행)')
    print('크기       기존 2차원(ms)  최적화 1차원(ms)  개선율(%)  연산 횟수(N²)')
    results = []
    for n in sizes:
        p, k = samples[n] if samples and n in samples else generate_patterns(n)
        flat_p, flat_k = flatten_matrix(p), flatten_matrix(k)
        mac(p, k)  # warm-up, outside measured interval
        mac_flat(flat_p, flat_k)
        elapsed_2d = elapsed_1d = 0
        for iteration in range(repeats):
            if iteration % 2:
                start = time.perf_counter_ns()
                mac_flat(flat_p, flat_k)
                elapsed_1d += time.perf_counter_ns() - start
                start = time.perf_counter_ns()
                mac(p, k)
                elapsed_2d += time.perf_counter_ns() - start
            else:
                start = time.perf_counter_ns()
                mac(p, k)
                elapsed_2d += time.perf_counter_ns() - start
                start = time.perf_counter_ns()
                mac_flat(flat_p, flat_k)
                elapsed_1d += time.perf_counter_ns() - start
        average_2d = elapsed_2d / repeats / 1_000_000
        average_1d = elapsed_1d / repeats / 1_000_000
        improvement = (1 - average_1d / average_2d) * 100 if average_2d else 0
        print('{:2}×{:<2}      {:.6f}        {:.6f}          {:+.1f}       {}'.format(
            n, n, average_2d, average_1d, improvement, n * n))
        results.append((n, average_2d, average_1d, n * n))
    return results


def read_matrix(title, n=3):
    print('\n{} ({}줄, 공백 구분)'.format(title, n))
    rows = []
    while len(rows) < n:
        try:
            row = [float(v) for v in input('{}행: '.format(len(rows) + 1)).split()]
            validate_matrix([row] * n, n)
        except ValueError:
            print('입력 형식 오류: 각 줄에 {}개의 유한한 숫자를 공백으로 구분해 입력하세요.'.format(n))
            continue
        rows.append(row)
    print('{} 저장 완료: {}'.format(title, rows))
    return rows


def interactive(repeats):
    a = read_matrix('필터 A')
    b = read_matrix('필터 B')
    pattern = read_matrix('패턴')
    flat_a, flat_b, flat_pattern = map(flatten_matrix, (a, b, pattern))
    start = time.perf_counter_ns()
    sa, sb = mac_flat(flat_pattern, flat_a), mac_flat(flat_pattern, flat_b)
    elapsed = (time.perf_counter_ns() - start) / 1_000_000
    print('\nA 점수: {:.17g}\nB 점수: {:.17g}'.format(sa, sb))
    result = decide(sa, sb, ('A', 'B'))
    print('판정: {}'.format('판정 불가 (UNDECIDED)' if result == 'UNDECIDED' else result))
    print('두 MAC 호출 시간: {:.6f} ms'.format(elapsed))
    benchmark([3], repeats, {3: (pattern, a)})


def analyze(data, repeats=100):
    if not isinstance(data, dict):
        raise ValueError('JSON 최상위는 객체여야 합니다.')
    filters = data.get('filters')
    patterns = data.get('patterns')
    if not isinstance(filters, dict) or not isinstance(patterns, dict):
        raise ValueError('filters와 patterns는 객체여야 합니다.')
    loaded, errors = {}, {}
    print('\n[필터 로드]')
    keys = set(filters) | {'size_5', 'size_13', 'size_25'}
    for key in sorted(keys):
        try:
            match = re.fullmatch(r'size_([1-9]\d*)', key)
            if not match:
                raise ValueError('필터 키 형식 오류')
            n = int(match.group(1))
            group = filters.get(key)
            if not isinstance(group, dict):
                raise ValueError('필터 그룹 누락 또는 객체 형식 오류')
            normalized = {}
            for label, matrix in group.items():
                canonical = normalize_label(label)
                if canonical in normalized:
                    raise ValueError('중복 표준 라벨: ' + canonical)
                normalized[canonical] = flatten_matrix(validate_matrix(matrix, n))
            if set(normalized) != {'Cross', 'X'}:
                raise ValueError('Cross와 X 필터가 모두 필요합니다.')
            loaded[key] = normalized
            print('{} 로드 완료 (Cross, X)'.format(key))
        except (ValueError, TypeError) as exc:
            errors[key] = str(exc)
            print('{} 필터 오류: {}'.format(key, exc))
    failures, passed = [], 0
    print('\n[패턴 분석]')
    for key, case in patterns.items():
        try:
            match = re.fullmatch(r'size_([1-9]\d*)_([1-9]\d*)', key)
            if not match:
                raise ValueError('패턴 키 형식은 size_{N}_{idx}여야 합니다.')
            n = int(match.group(1))
            filter_key = 'size_{}'.format(n)
            if filter_key not in loaded:
                raise ValueError('필터 사용 불가: ' + errors.get(filter_key, filter_key + ' 누락'))
            if not isinstance(case, dict):
                raise ValueError('패턴 항목은 객체여야 합니다.')
            expected = normalize_label(case.get('expected'))
            pattern = flatten_matrix(validate_matrix(case.get('input'), n))
            group = loaded[filter_key]
            a, b = mac_flat(pattern, group['Cross']), mac_flat(pattern, group['X'])
            result = decide(a, b)
            ok = result == expected
            print('{}: Cross={:.17g}, X={:.17g} | 판정={} | expected={} | {}'.format(
                key, a, b, result, expected, 'PASS' if ok else 'FAIL'))
            if ok:
                passed += 1
            else:
                reason = '동점: |Cross-X| < 1e-9' if result == 'UNDECIDED' else '예상 라벨과 판정 불일치'
                failures.append((key, reason))
        except (ValueError, TypeError, OverflowError) as exc:
            failures.append((key, str(exc)))
            print('{}: FAIL | {}'.format(key, exc))
    benchmark([3, 5, 13, 25], repeats)
    print('\n[결과 요약]\n총 테스트: {}개\n통과: {}개\n실패: {}개'.format(len(patterns), passed, len(failures)))
    if not patterns:
        print('주의: 패턴이 없어 검증된 케이스가 없습니다.')
    for key, reason in failures:
        print('- {}: {}'.format(key, reason))
    return passed, failures


def main():
    parser = argparse.ArgumentParser(description='Mini NPU Simulator')
    parser.add_argument('--mode', choices=('1', '2'))
    parser.add_argument('--data', type=Path, default=Path(__file__).with_name('data.json'))
    parser.add_argument('--repeats', type=int, default=100)
    args = parser.parse_args()
    if args.repeats < 10:
        parser.error('--repeats는 10 이상이어야 합니다.')
    print('=== Mini NPU Simulator ===')
    try:
        mode = args.mode
        while mode not in ('1', '2'):
            print('\n1. 사용자 입력 (3×3)\n2. data.json 분석')
            mode = input('선택: ').strip()
            if mode not in ('1', '2'):
                print('1 또는 2를 입력하세요.')
        if mode == '1':
            interactive(args.repeats)
            return 0
        with args.data.open(encoding='utf-8-sig') as stream:
            data = json.load(stream)
        _, failures = analyze(data, args.repeats)
        return 1 if failures or not data['patterns'] else 0
    except (OSError, ValueError, OverflowError) as exc:
        print('오류: {}'.format(exc))
        return 2
    except (EOFError, KeyboardInterrupt):
        print('\n입력이 종료되어 프로그램을 마칩니다.')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
