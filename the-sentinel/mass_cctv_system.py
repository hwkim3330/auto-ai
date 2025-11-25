#!/usr/bin/env python3
"""
Mass CCTV Monitoring System
============================

서울 전역 5,000+ CCTV 동시 모니터링 시스템
Person of Interest 스타일 - 모든 카메라 동시 처리

Features:
- 전체 CCTV 목록 가져오기
- 지역별 필터링 (강남, 종로, 마포 등)
- 멀티스레드 동시 처리 (수백 개)
- 실시간 지도에 모두 표시
- 우선순위 기반 처리
"""

import cv2
import numpy as np
import requests
import json
import time
import threading
import queue
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import defaultdict
import re


@dataclass
class CCTVInfo:
    """CCTV 정보"""
    id: str
    name: str
    location: str
    latitude: float
    longitude: float
    district: str  # 구 (강남구, 종로구 등)
    stream_url: Optional[str] = None
    status: str = "idle"  # idle, processing, active, error
    last_update: Optional[float] = None


class CCTVRegistry:
    """
    CCTV 레지스트리 - 전체 목록 관리

    실제로는 TOPIS API에서 가져오지만,
    여기서는 패턴 기반 생성 + 실제 발견된 URL 사용
    """

    def __init__(self):
        self.all_cctvs: Dict[str, CCTVInfo] = {}
        self.by_district: Dict[str, List[str]] = defaultdict(list)

    def load_from_topis_api(self) -> int:
        """
        TOPIS API에서 전체 CCTV 목록 가져오기

        실제 API 엔드포인트 (발견 필요):
        - GET https://topis.seoul.go.kr/api/cctvList
        - 또는 F12 Network 탭에서 발견한 엔드포인트
        """

        # 실제 API 시도
        api_urls = [
            "https://topis.seoul.go.kr/api/cctvList",
            "https://topis.seoul.go.kr/data/getCctvList.do",
            "https://topis.seoul.go.kr/map/getCctvInfoAll.do"
        ]

        for url in api_urls:
            try:
                response = requests.get(
                    url,
                    headers={
                        'User-Agent': 'Mozilla/5.0',
                        'Referer': 'https://topis.seoul.go.kr/map/openCctvMap.do'
                    },
                    timeout=5
                )

                if response.status_code == 200:
                    data = response.json()
                    print(f"[Success] Found API: {url}")
                    return self._parse_api_response(data)

            except Exception as e:
                continue

        # API 못 찾으면 샘플 데이터 생성
        print("[Info] API not found, using sample data")
        return self._generate_sample_data()

    def _parse_api_response(self, data: dict) -> int:
        """API 응답 파싱"""
        # 실제 응답 구조에 맞게 수정 필요
        # 예상 구조: { "cctvList": [...] } 또는 [...] 직접

        cctvs = data.get('cctvList', data.get('data', data))

        for item in cctvs:
            cctv = CCTVInfo(
                id=item.get('cctvId', item.get('id')),
                name=item.get('cctvName', item.get('name')),
                location=item.get('location', item.get('address')),
                latitude=float(item.get('latitude', item.get('lat', 0))),
                longitude=float(item.get('longitude', item.get('lon', 0))),
                district=item.get('district', self._extract_district(item.get('location', ''))),
                stream_url=item.get('streamUrl', item.get('url'))
            )

            self.all_cctvs[cctv.id] = cctv
            self.by_district[cctv.district].append(cctv.id)

        return len(self.all_cctvs)

    def _generate_sample_data(self) -> int:
        """
        샘플 데이터 생성

        서울 25개 구 × 200개 CCTV = 5,000개
        실제로는 API에서 가져와야 함
        """

        # 서울 25개 구
        districts = {
            '강남구': (37.4979, 127.0276),
            '강동구': (37.5301, 127.1238),
            '강북구': (37.6396, 127.0254),
            '강서구': (37.5509, 126.8495),
            '관악구': (37.4784, 126.9516),
            '광진구': (37.5384, 127.0822),
            '구로구': (37.4955, 126.8874),
            '금천구': (37.4519, 126.9020),
            '노원구': (37.6543, 127.0567),
            '도봉구': (37.6688, 127.0471),
            '동대문구': (37.5744, 127.0399),
            '동작구': (37.5124, 126.9393),
            '마포구': (37.5663, 126.9019),
            '서대문구': (37.5791, 126.9368),
            '서초구': (37.4837, 127.0324),
            '성동구': (37.5634, 127.0371),
            '성북구': (37.5894, 127.0167),
            '송파구': (37.5145, 127.1059),
            '양천구': (37.5170, 126.8664),
            '영등포구': (37.5264, 126.8963),
            '용산구': (37.5324, 126.9909),
            '은평구': (37.6027, 126.9292),
            '종로구': (37.5735, 126.9790),
            '중구': (37.5640, 126.9970),
            '중랑구': (37.6063, 127.0925),
        }

        count = 0

        for district, (base_lat, base_lon) in districts.items():
            # 각 구마다 200개 CCTV
            for i in range(200):
                # 랜덤 오프셋 (약 5km 반경)
                lat_offset = (np.random.random() - 0.5) * 0.05
                lon_offset = (np.random.random() - 0.5) * 0.05

                cctv_id = f"{district[:2]}{i+1:04d}"

                cctv = CCTVInfo(
                    id=cctv_id,
                    name=f"{district} CCTV {i+1}",
                    location=f"{district} {['역삼동', '서초동', '방배동', '개포동'][i % 4]}",
                    latitude=base_lat + lat_offset,
                    longitude=base_lon + lon_offset,
                    district=district,
                    stream_url=None  # 실제로는 발견 필요
                )

                self.all_cctvs[cctv_id] = cctv
                self.by_district[district].append(cctv_id)
                count += 1

        return count

    def _extract_district(self, location: str) -> str:
        """주소에서 구 추출"""
        districts = ['강남구', '강동구', '강북구', '강서구', '관악구',
                    '광진구', '구로구', '금천구', '노원구', '도봉구',
                    '동대문구', '동작구', '마포구', '서대문구', '서초구',
                    '성동구', '성북구', '송파구', '양천구', '영등포구',
                    '용산구', '은평구', '종로구', '중구', '중랑구']

        for district in districts:
            if district in location:
                return district

        return '기타'

    def get_by_district(self, district: str) -> List[CCTVInfo]:
        """구별 CCTV 목록"""
        return [self.all_cctvs[cctv_id] for cctv_id in self.by_district[district]]

    def get_by_area(self, center_lat: float, center_lon: float, radius_km: float = 2.0) -> List[CCTVInfo]:
        """반경 내 CCTV 검색"""
        results = []

        for cctv in self.all_cctvs.values():
            # 간단한 거리 계산 (정확하지 않지만 빠름)
            lat_diff = (cctv.latitude - center_lat) * 111  # km
            lon_diff = (cctv.longitude - center_lon) * 88.8  # km (위도 37도 기준)
            distance = np.sqrt(lat_diff**2 + lon_diff**2)

            if distance <= radius_km:
                results.append(cctv)

        return results


class MultiCCTVProcessor:
    """
    다중 CCTV 동시 처리 시스템

    멀티스레드로 수백 개 CCTV 동시 처리
    우선순위 기반 스케줄링
    """

    def __init__(self, max_workers: int = 50):
        self.max_workers = max_workers
        self.registry = CCTVRegistry()
        self.active_streams: Dict[str, cv2.VideoCapture] = {}
        self.frame_queues: Dict[str, queue.Queue] = {}
        self.stop_event = threading.Event()

        # 통계
        self.stats = {
            'total_cctvs': 0,
            'active': 0,
            'errors': 0,
            'frames_processed': 0
        }

    def load_all_cctvs(self):
        """전체 CCTV 목록 로드"""
        count = self.registry.load_from_topis_api()
        self.stats['total_cctvs'] = count
        print(f"[Registry] Loaded {count:,} CCTVs")

        # 구별 통계
        print("\n구별 CCTV 수:")
        for district, cctv_ids in sorted(self.registry.by_district.items()):
            print(f"  {district}: {len(cctv_ids):,}개")

    def start_monitoring(self, districts: Optional[List[str]] = None, max_cctvs: int = 100):
        """
        모니터링 시작

        Args:
            districts: 모니터링할 구 목록 (None = 전체)
            max_cctvs: 최대 동시 처리 CCTV 수
        """

        # 모니터링할 CCTV 선택
        if districts:
            cctvs_to_monitor = []
            for district in districts:
                cctvs_to_monitor.extend(self.registry.get_by_district(district))
        else:
            cctvs_to_monitor = list(self.registry.all_cctvs.values())

        # 최대 개수 제한
        cctvs_to_monitor = cctvs_to_monitor[:max_cctvs]

        print(f"\n[Monitoring] Starting {len(cctvs_to_monitor)} CCTVs with {self.max_workers} workers")

        # ThreadPoolExecutor로 동시 처리
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []

            for cctv in cctvs_to_monitor:
                future = executor.submit(self._process_cctv, cctv)
                futures.append(future)

            # 진행 상황 모니터링
            try:
                while not all(f.done() for f in futures):
                    active = sum(1 for f in futures if not f.done())
                    print(f"\r[Status] Active: {active}/{len(futures)} | "
                          f"Frames: {self.stats['frames_processed']} | "
                          f"Errors: {self.stats['errors']}", end='')
                    time.sleep(1)

            except KeyboardInterrupt:
                print("\n[Stop] Shutting down...")
                self.stop_event.set()

    def _process_cctv(self, cctv: CCTVInfo):
        """개별 CCTV 처리 (스레드에서 실행)"""

        # 스트림 URL 없으면 스킵
        if not cctv.stream_url:
            # 실제로는 여기서 URL 발견 시도
            # 예: topis_stream_capture 사용
            cctv.status = "no_url"
            return

        try:
            # 스트림 연결
            cap = cv2.VideoCapture(cctv.stream_url)

            if not cap.isOpened():
                cctv.status = "error"
                self.stats['errors'] += 1
                return

            cctv.status = "active"
            self.stats['active'] += 1

            # 프레임 처리 (계속 또는 샘플링)
            frame_count = 0

            while not self.stop_event.is_set():
                ret, frame = cap.read()

                if not ret:
                    break

                # 프레임 처리 (객체 탐지는 나중에)
                # 지금은 통계만
                frame_count += 1
                self.stats['frames_processed'] += 1

                # 5초에 1프레임만 처리 (부하 감소)
                time.sleep(5)

            cap.release()
            cctv.status = "stopped"

        except Exception as e:
            cctv.status = "error"
            self.stats['errors'] += 1
            print(f"\n[Error] {cctv.id}: {e}")

    def get_statistics(self) -> Dict:
        """통계 반환"""
        return {
            **self.stats,
            'by_district': {
                district: len(cctv_ids)
                for district, cctv_ids in self.registry.by_district.items()
            }
        }


class SmartSelector:
    """
    스마트 CCTV 선택기

    모든 CCTV를 처리할 수 없으므로 우선순위 기반 선택
    """

    def __init__(self, registry: CCTVRegistry):
        self.registry = registry

    def select_by_priority(self, max_count: int = 100) -> List[CCTVInfo]:
        """
        우선순위 기반 선택

        우선순위:
        1. 주요 교차로 (강남역, 광화문 등)
        2. 교통 혼잡 지역
        3. 골고루 분포 (각 구에서 일부씩)
        """

        selected = []

        # 1. 주요 지점 (이름으로 판단)
        keywords = ['역', '광장', '사거리', '교차로', '터미널']

        for cctv in self.registry.all_cctvs.values():
            if any(kw in cctv.name for kw in keywords):
                selected.append(cctv)

                if len(selected) >= max_count // 2:
                    break

        # 2. 각 구에서 균등하게
        per_district = (max_count - len(selected)) // len(self.registry.by_district)

        for district, cctv_ids in self.registry.by_district.items():
            for cctv_id in cctv_ids[:per_district]:
                selected.append(self.registry.all_cctvs[cctv_id])

        return selected[:max_count]


def main():
    print("=" * 70)
    print("MASS CCTV MONITORING SYSTEM")
    print("서울 전역 5,000+ CCTV 동시 모니터링")
    print("=" * 70)

    # 시스템 초기화
    processor = MultiCCTVProcessor(max_workers=50)

    # 전체 CCTV 로드
    processor.load_all_cctvs()

    print("\n옵션:")
    print("1. 전체 모니터링 (5,000개+)")
    print("2. 특정 구만 (예: 강남구, 종로구)")
    print("3. 스마트 선택 (우선순위 100개)")
    print("4. 반경 검색 (특정 지점 주변)")

    choice = input("\n선택 (1-4): ").strip()

    if choice == '1':
        # 전체 (실제로는 제한 필요)
        processor.start_monitoring(max_cctvs=100)

    elif choice == '2':
        # 특정 구
        districts = input("구 이름 입력 (쉼표로 구분): ").split(',')
        districts = [d.strip() for d in districts]
        processor.start_monitoring(districts=districts, max_cctvs=100)

    elif choice == '3':
        # 스마트 선택
        selector = SmartSelector(processor.registry)
        selected = selector.select_by_priority(max_count=100)

        print(f"\n[Smart] Selected {len(selected)} priority CCTVs")
        # TODO: 선택된 CCTV만 모니터링

    elif choice == '4':
        # 반경 검색
        lat = float(input("위도: "))
        lon = float(input("경도: "))
        radius = float(input("반경 (km): "))

        cctvs = processor.registry.get_by_area(lat, lon, radius)
        print(f"\n[Area] Found {len(cctvs)} CCTVs in {radius}km radius")
        # TODO: 검색된 CCTV만 모니터링

    # 최종 통계
    stats = processor.get_statistics()

    print("\n\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)
    print(f"Total CCTVs: {stats['total_cctvs']:,}")
    print(f"Active: {stats['active']}")
    print(f"Errors: {stats['errors']}")
    print(f"Frames Processed: {stats['frames_processed']:,}")

    print("\n구별 분포:")
    for district, count in sorted(stats['by_district'].items()):
        print(f"  {district}: {count:,}개")


if __name__ == "__main__":
    main()
