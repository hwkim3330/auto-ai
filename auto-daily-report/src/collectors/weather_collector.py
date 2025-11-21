"""
Weather Collector

Collects weather data from OpenWeatherMap API
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class WeatherCollector:
    """날씨 정보 수집기"""

    def __init__(self, config: dict):
        """
        Initialize Weather Collector

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config.get('weather', {})
        self.api_key = os.getenv('OPENWEATHER_API_KEY')
        self.city = self.config.get('city', 'Seoul')
        self.country_code = self.config.get('country_code', 'KR')
        self.units = self.config.get('units', 'metric')
        self.forecast_days = self.config.get('forecast_days', 3)
        self.include_aqi = self.config.get('include_aqi', True)

    def collect_all(self) -> Dict:
        """
        모든 날씨 정보 수집

        Returns:
            날씨 정보 딕셔너리
        """
        if not self.api_key:
            logger.error("OpenWeatherMap API key not found!")
            return {'error': 'API key not configured'}

        weather_data = {
            'current': None,
            'forecast': None,
            'air_quality': None,
            'alerts': [],
            'clothing_advice': None,
            'collected_at': datetime.utcnow().isoformat()
        }

        # 현재 날씨
        logger.info(f"Collecting current weather for {self.city}...")
        weather_data['current'] = self._get_current_weather()

        # 예보
        logger.info(f"Collecting {self.forecast_days}-day forecast...")
        weather_data['forecast'] = self._get_forecast()

        # 대기질
        if self.include_aqi and weather_data['current']:
            logger.info("Collecting air quality data...")
            lat = weather_data['current'].get('lat')
            lon = weather_data['current'].get('lon')
            if lat and lon:
                weather_data['air_quality'] = self._get_air_quality(lat, lon)

        # 날씨 경고 생성
        weather_data['alerts'] = self._generate_alerts(weather_data)

        # 옷차림 추천
        if weather_data['current']:
            weather_data['clothing_advice'] = self._get_clothing_advice(
                weather_data['current'].get('temp')
            )

        logger.info("Weather data collection completed")
        return weather_data

    def _get_current_weather(self) -> Optional[Dict]:
        """현재 날씨 조회"""
        try:
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                'q': f"{self.city},{self.country_code}",
                'appid': self.api_key,
                'units': self.units,
                'lang': 'kr'
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            return {
                'city': data['name'],
                'lat': data['coord']['lat'],
                'lon': data['coord']['lon'],
                'temp': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'temp_min': data['main']['temp_min'],
                'temp_max': data['main']['temp_max'],
                'pressure': data['main']['pressure'],
                'humidity': data['main']['humidity'],
                'weather': data['weather'][0]['main'],
                'weather_description': data['weather'][0]['description'],
                'weather_icon': data['weather'][0']['icon'],
                'clouds': data['clouds']['all'],
                'wind_speed': data['wind']['speed'],
                'wind_deg': data.get('wind', {}).get('deg'),
                'visibility': data.get('visibility'),
                'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).isoformat(),
                'sunset': datetime.fromtimestamp(data['sys']['sunset']).isoformat(),
                'timezone': data['timezone']
            }

        except Exception as e:
            logger.error(f"Error fetching current weather: {e}")
            return None

    def _get_forecast(self) -> Optional[List[Dict]]:
        """일기예보 조회 (5-day/3-hour)"""
        try:
            url = "https://api.openweathermap.org/data/2.5/forecast"
            params = {
                'q': f"{self.city},{self.country_code}",
                'appid': self.api_key,
                'units': self.units,
                'lang': 'kr',
                'cnt': self.forecast_days * 8  # 3-hour intervals, 8 per day
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            forecasts = []
            for item in data['list']:
                forecasts.append({
                    'datetime': item['dt_txt'],
                    'temp': item['main']['temp'],
                    'feels_like': item['main']['feels_like'],
                    'temp_min': item['main']['temp_min'],
                    'temp_max': item['main']['temp_max'],
                    'pressure': item['main']['pressure'],
                    'humidity': item['main']['humidity'],
                    'weather': item['weather'][0]['main'],
                    'weather_description': item['weather'][0]['description'],
                    'weather_icon': item['weather'][0]['icon'],
                    'clouds': item['clouds']['all'],
                    'wind_speed': item['wind']['speed'],
                    'rain_3h': item.get('rain', {}).get('3h', 0),
                    'snow_3h': item.get('snow', {}).get('3h', 0),
                    'pop': item.get('pop', 0)  # Probability of precipitation
                })

            # Daily summary (first forecast of each day)
            daily_forecasts = []
            seen_dates = set()
            for fc in forecasts:
                date = fc['datetime'].split()[0]
                if date not in seen_dates:
                    daily_forecasts.append(fc)
                    seen_dates.add(date)

            return daily_forecasts[:self.forecast_days]

        except Exception as e:
            logger.error(f"Error fetching forecast: {e}")
            return None

    def _get_air_quality(self, lat: float, lon: float) -> Optional[Dict]:
        """대기질 정보 조회"""
        try:
            url = "http://api.openweathermap.org/data/2.5/air_pollution"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data['list']:
                aqi_data = data['list'][0]
                components = aqi_data['components']

                # AQI 수치 (1=Good, 2=Fair, 3=Moderate, 4=Poor, 5=Very Poor)
                aqi_index = aqi_data['main']['aqi']
                aqi_labels = {1: 'Good', 2: 'Fair', 3: 'Moderate', 4: 'Poor', 5: 'Very Poor'}
                aqi_labels_kr = {1: '좋음', 2: '보통', 3: '나쁨', 4: '매우 나쁨', 5: '최악'}

                return {
                    'aqi': aqi_index,
                    'aqi_label': aqi_labels.get(aqi_index, 'Unknown'),
                    'aqi_label_kr': aqi_labels_kr.get(aqi_index, '알 수 없음'),
                    'pm2_5': components.get('pm2_5'),
                    'pm10': components.get('pm10'),
                    'o3': components.get('o3'),
                    'no2': components.get('no2'),
                    'so2': components.get('so2'),
                    'co': components.get('co')
                }

        except Exception as e:
            logger.error(f"Error fetching air quality: {e}")
            return None

    def _generate_alerts(self, weather_data: Dict) -> List[str]:
        """날씨 경고 생성"""
        alerts = []

        current = weather_data.get('current')
        forecast = weather_data.get('forecast', [])
        air_quality = weather_data.get('air_quality')

        if not current:
            return alerts

        # 온도 경고
        temp_alerts = self.config.get('alerts', {}).get('temperature_alerts', {})
        temp = current.get('temp')

        if temp and temp < temp_alerts.get('cold', 0):
            alerts.append(f"⚠️ 한파 주의: 기온 {temp}°C")
        elif temp and temp > temp_alerts.get('hot', 35):
            alerts.append(f"⚠️ 폭염 주의: 기온 {temp}°C")

        # 강수 확률 경고
        rain_threshold = self.config.get('alerts', {}).get('rain_probability_threshold', 70)
        if forecast:
            for fc in forecast[:3]:  # 오늘 예보만
                if fc.get('pop', 0) * 100 > rain_threshold:
                    alerts.append(f"☔ 강수 예상: 확률 {fc['pop']*100:.0f}%")
                    break

        # 대기질 경고
        if air_quality:
            aqi_threshold = self.config.get('alerts', {}).get('aqi_threshold', 100)
            if air_quality.get('pm2_5', 0) > aqi_threshold:
                alerts.append(f"🌫️ 미세먼지 주의: PM2.5 {air_quality['pm2_5']} µg/m³")

        return alerts

    def _get_clothing_advice(self, temp: Optional[float]) -> Optional[str]:
        """옷차림 추천"""
        if temp is None:
            return None

        clothing_config = self.config.get('clothing_advice', {})
        if not clothing_config.get('enabled', True):
            return None

        thresholds = clothing_config.get('thresholds', {})

        if temp < thresholds.get('cold', 5):
            return "두꺼운 외투, 목도리, 장갑 착용 권장"
        elif temp < thresholds.get('cool', 15):
            return "가을 자켓 또는 가디건 권장"
        elif temp < thresholds.get('mild', 20):
            return "긴팔 셔츠 또는 얇은 가디건"
        elif temp < thresholds.get('warm', 25):
            return "반팔 또는 얇은 긴팔"
        else:
            return "시원한 여름 옷차림, 자외선 차단 필수"


if __name__ == '__main__':
    # Test
    import yaml

    logging.basicConfig(level=logging.INFO)

    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    collector = WeatherCollector(config)
    weather = collector.collect_all()

    print("\n=== Weather Collection Test ===")
    if weather.get('current'):
        curr = weather['current']
        print(f"\n현재 날씨 ({curr['city']}):")
        print(f"  온도: {curr['temp']}°C (체감: {curr['feels_like']}°C)")
        print(f"  날씨: {curr['weather_description']}")
        print(f"  습도: {curr['humidity']}%")
        print(f"  풍속: {curr['wind_speed']} m/s")

    if weather.get('air_quality'):
        aqi = weather['air_quality']
        print(f"\n대기질:")
        print(f"  AQI: {aqi['aqi_label_kr']} (PM2.5: {aqi['pm2_5']} µg/m³)")

    if weather.get('clothing_advice'):
        print(f"\n옷차림: {weather['clothing_advice']}")

    if weather.get('alerts'):
        print(f"\n경고:")
        for alert in weather['alerts']:
            print(f"  {alert}")
