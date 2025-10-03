 # weather APIs, holiday calendar etc

import os
import requests
import random
import logging
import calendar as cal
from src.logger import logger
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)

class WeatherAPI:
    def __init__(self, api_key=None):
        # Use env var or fallback to demo key
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY', 'demo_key')
        self.base_url = "https://api.openweathermap.org/data/2.5"
        logger.info("WeatherAPI initialized.")
        
    def get_current_weather(self, store_ids):
        """Get current weather for a list of stores."""
        weather_data = {}
        logger.info(f"Fetching current weather for {len(store_ids)} stores.")
        
        for store_id in store_ids:
            try:
                # For production, replace the below block with actual API call
                # Example:
                # response = requests.get(
                #     f"{self.base_url}/weather",
                #     params={"id": store_id, "appid": self.api_key, "units": "metric"}
                # )
                # response.raise_for_status()
                # data = response.json()
                # Parse real data here
                
                # Simulated data for demo purposes:
                conditions = ['clear', 'rain', 'cloudy']
                weather_data[store_id] = {
                    'temperature': random.randint(15, 35),
                    'humidity': random.randint(30, 80),
                    'wind_speed': random.randint(5, 25),
                    'visibility': random.randint(5, 15),
                    'weather_condition': random.choice(conditions),
                    'pressure': random.randint(1000, 1020)
                }
                logger.debug(f"Weather for store {store_id}: {weather_data[store_id]}")
            
            except requests.RequestException as e:
                logger.error(f"Failed to fetch weather for store {store_id}: {e}")
                # Decide fallback: maybe default weather or skip
                weather_data[store_id] = None
        
        return weather_data
    
    def get_weather_forecast(self, store_ids, days):
        """Get weather forecast for stores for given number of days."""
        forecast_data = {}
        logger.info(f"Fetching weather forecast for {len(store_ids)} stores for {days} days.")
        
        for store_id in store_ids:
            try:
                # For production, replace below with real API call
                # Example endpoint: forecast/daily or onecall API
                
                forecast_data[store_id] = []
                for i in range(days):
                    conditions = ['clear', 'rain', 'cloudy']
                    forecast_data[store_id].append({
                        'date': (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d'),
                        'temperature': random.randint(15, 35),
                        'humidity': random.randint(30, 80),
                        'wind_speed': random.randint(5, 25),
                        'weather_condition': random.choice(conditions)
                    })
                logger.debug(f"Forecast for store {store_id}: {forecast_data[store_id]}")
            
            except requests.RequestException as e:
                logger.error(f"Failed to fetch forecast for store {store_id}: {e}")
                forecast_data[store_id] = None
        
        return forecast_data



class HolidayAPI:
    def __init__(self):
        logger.info("Initializing HolidayAPI for 2025 with dynamic holidays.")
        self.holidays = self._generate_holidays_2025()
    
    def _nth_weekday_of_month(self, year, month, weekday, n):
        """Return date string for nth weekday of a month (e.g., 4th Thursday Nov)"""
        count = 0
        for day in range(1, 32):
            try:
                date = datetime(year, month, day)
            except ValueError:
                break  # End of month
            if date.weekday() == weekday:
                count += 1
                if count == n:
                    return date.strftime('%Y-%m-%d')
        return None
    
    def _last_weekday_of_month(self, year, month, weekday):
        """Return date string for last weekday of a month (e.g., last Monday of May)"""
        last_day = cal.monthrange(year, month)[1]
        for day in range(last_day, 0, -1):
            date = datetime(year, month, day)
            if date.weekday() == weekday:
                return date.strftime('%Y-%m-%d')
        return None
    
    def _generate_holidays_2025(self):
        year = 2025
        
        holidays = {
            'New Year\'s Day': {
                'date': f'{year}-01-01',
                'type': 'major',
                'impact': {'SKU001': 2.5, 'SKU002': 1.4, 'SKU003': 1.8, 'SKU004': 1.2, 'SKU005': 2.7}
            },
            'Memorial Day': {
                'date': self._last_weekday_of_month(year, 5, 0),  # last Monday of May
                'type': 'major',
                'impact': {'SKU001': 2.0, 'SKU002': 1.5, 'SKU003': 1.7, 'SKU004': 1.4, 'SKU005': 2.5}
            },
            'Independence Day': {
                'date': f'{year}-07-04',
                'type': 'major',
                'impact': {'SKU001': 1.6, 'SKU002': 1.4, 'SKU003': 1.5, 'SKU004': 1.3, 'SKU005': 2.0}
            },
            'Labor Day': {
                'date': self._nth_weekday_of_month(year, 9, 0, 1),  # first Monday of Sept
                'type': 'major',
                'impact': {'SKU001': 1.8, 'SKU002': 1.3, 'SKU003': 1.6, 'SKU004': 1.4, 'SKU005': 2.2}
            },
            'Thanksgiving': {
                'date': self._nth_weekday_of_month(year, 11, 3, 4),  # 4th Thursday Nov
                'type': 'major',
                'impact': {'SKU001': 2.2, 'SKU002': 1.6, 'SKU003': 2.5, 'SKU004': 1.9, 'SKU005': 3.2}
            },
            'Black Friday': {
                # Day after Thanksgiving
                'date': (datetime.strptime(self._nth_weekday_of_month(year, 11, 3, 4), '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d'),
                'type': 'major',
                'impact': {'SKU001': 2.5, 'SKU002': 1.8, 'SKU003': 2.0, 'SKU004': 1.5, 'SKU005': 2.8}
            },
            'Christmas': {
                'date': f'{year}-12-25',
                'type': 'major',
                'impact': {'SKU001': 2.8, 'SKU002': 1.5, 'SKU003': 2.2, 'SKU004': 1.8, 'SKU005': 3.0}
            },
            'New Year\'s Eve': {
                'date': f'{year}-12-31',
                'type': 'major',
                'impact': {'SKU001': 2.0, 'SKU002': 1.3, 'SKU003': 1.6, 'SKU004': 1.2, 'SKU005': 2.5}
            },
            'Valentine\'s Day': {
                'date': f'{year}-02-14',
                'type': 'minor',
                'impact': {'SKU001': 1.8, 'SKU002': 1.4, 'SKU003': 1.3, 'SKU004': 1.1, 'SKU005': 1.9}
            }
        }
        logger.info(f"Generated holidays for {year}: {holidays}")
        # Convert to internal dict with date strings as keys for fast lookup
        holidays_by_date = {}
        for name, info in holidays.items():
            holidays_by_date[info['date']] = {'name': name, **info}
        return holidays_by_date
    
    def get_holidays(self, days):
        logger.info(f"Fetching holidays for next {days} days.")
        holidays = []
        today = datetime.now()
        for i in range(days):
            date_str = (today + timedelta(days=i)).strftime('%Y-%m-%d')
            if date_str in self.holidays:
                holidays.append(self.holidays[date_str])
        return holidays
    
    def get_holiday_impact(self, sku_id, date):
        date_str = date.strftime('%Y-%m-%d') if isinstance(date, datetime) else date
        impact = self.holidays.get(date_str, {}).get('impact', {}).get(sku_id, 1.0)
        logger.debug(f"Holiday impact for SKU {sku_id} on {date_str}: {impact}")
        return impact
    
    def get_upcoming_holidays(self, days=30):
        logger.info(f"Fetching upcoming holidays for next {days} days.")
        upcoming = []
        today = datetime.now()
        for i in range(days):
            date_str = (today + timedelta(days=i)).strftime('%Y-%m-%d')
            if date_str in self.holidays:
                holiday = self.holidays[date_str].copy()
                holiday['days_until'] = i
                upcoming.append(holiday)
        return upcoming
