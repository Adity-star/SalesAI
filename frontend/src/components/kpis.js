import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Card,
  CardContent,
  Grid,
  Chip,
  CircularProgress
} from '@mui/material';
import {
  WbSunny,
  Cloud,
  Opacity,
  Air,
  Event,
  TrendingUp
} from '@mui/icons-material';
import { fetchWeatherData } from '../services/api';

const WeatherWidget = ({ selectedStores = [], selectedSKUs = [] }) => {
  const [weatherData, setWeatherData] = useState(null);
  const [loading, setLoading] = useState(false);

  // Default stores to show if none selected
  const displayStores = selectedStores.length > 0 ? selectedStores : ['STORE001', 'STORE002', 'STORE003'];
  // Default SKUs to show if none selected
  const displaySKUs = selectedSKUs.length > 0 ? selectedSKUs : ['SKU001', 'SKU002', 'SKU003', 'SKU004', 'SKU005'];

  useEffect(() => {
    loadWeatherData();
  }, [displayStores]);

  const loadWeatherData = async () => {
    setLoading(true);
    try {
      const data = await fetchWeatherData(displayStores);
      setWeatherData(data);
    } catch (error) {
      console.error('Weather data error:', error);
      // Generate mock data if API fails
      setWeatherData({
        weather: {
          'STORE001': {
            temperature: 22,
            humidity: 65,
            wind_speed: 12,
            visibility: 10,
            weather_condition: 'clear',
            pressure: 1013
          },
          'STORE002': {
            temperature: 28,
            humidity: 45,
            wind_speed: 8,
            visibility: 15,
            weather_condition: 'clear',
            pressure: 1015
          },
          'STORE003': {
            temperature: 18,
            humidity: 75,
            wind_speed: 15,
            visibility: 8,
            weather_condition: 'rain',
            pressure: 1008
          }
        }
      });
    } finally {
      setLoading(false);
    }
  };

  const getWeatherIcon = (condition) => {
    switch (condition) {
      case 'clear': return <WbSunny />;
      case 'rain': return <Opacity />;
      case 'cloudy': return <Cloud />;
      default: return <WbSunny />;
    }
  };

  const getWeatherColor = (condition) => {
    switch (condition) {
      case 'clear': return 'success';
      case 'rain': return 'info';
      case 'cloudy': return 'warning';
      default: return 'default';
    }
  };

  // Holiday data with impact on different product categories
  const holidays = [
    {
      name: 'Christmas',
      date: 'Dec 25, 2024',
      impact: {
        'SKU001': 2.8, // Coffee beans - high demand
        'SKU002': 1.5, // Milk - moderate
        'SKU003': 2.2, // Bread - high
        'SKU004': 1.8, // Bananas - moderate
        'SKU005': 3.0  // Chicken - very high
      },
      color: '#e74c3c'
    },
    {
      name: 'New Year\'s Eve',
      date: 'Dec 31, 2024',
      impact: {
        'SKU001': 2.0, // Coffee - high
        'SKU002': 1.3, // Milk - low
        'SKU003': 1.6, // Bread - moderate
        'SKU004': 1.2, // Bananas - low
        'SKU005': 2.5  // Chicken - high
      },
      color: '#3498db'
    },
    {
      name: 'Valentine\'s Day',
      date: 'Feb 14, 2025',
      impact: {
        'SKU001': 1.8, // Coffee - moderate
        'SKU002': 1.4, // Milk - moderate
        'SKU003': 1.3, // Bread - low
        'SKU004': 1.1, // Bananas - low
        'SKU005': 1.9  // Chicken - moderate
      },
      color: '#e91e63'
    }
  ];

  const getHolidayImpact = (skuId) => {
    const upcomingHoliday = holidays[0]; // Get the next holiday
    return upcomingHoliday.impact[skuId] || 1.0;
  };

  const getImpactColor = (impact) => {
    if (impact >= 2.5) return 'error';
    if (impact >= 1.8) return 'warning';
    if (impact >= 1.3) return 'info';
    return 'success';
  };

  const getSKUName = (skuId) => {
    const skuNames = {
      'SKU001': 'Coffee Beans',
      'SKU002': 'Milk',
      'SKU003': 'Bread',
      'SKU004': 'Bananas',
      'SKU005': 'Chicken'
    };
    return skuNames[skuId] || skuId;
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <CircularProgress />
      </Box>
    );
  }

  if (!weatherData || !weatherData.weather || Object.keys(weatherData.weather).length === 0) {
    return (
      <Paper sx={{ 
        p: 4, 
        textAlign: 'center',
        background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
        border: '1px solid #333333',
        borderRadius: 4
      }}>
        <WbSunny sx={{ 
          fontSize: 60, 
          color: '#0071ce', 
          mb: 2,
          opacity: 0.6
        }} />
        <Typography variant="h6" sx={{ 
          fontWeight: 600,
          mb: 1
        }}>
          Weather Data Unavailable
        </Typography>
        <Typography variant="body2" sx={{ opacity: 0.7 }}>
          Select stores to view weather information
        </Typography>
      </Paper>
    );
  }

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        <WbSunny sx={{ mr: 1, verticalAlign: 'middle' }} />
        Weather & External Factors
      </Typography>

      <Grid container spacing={4}>
        {Object.entries(weatherData.weather).map(([storeId, weather]) => (
          <Grid item xs={12} md={6} lg={4} key={storeId}>
            <Card sx={{ height: '100%' }}>
              <CardContent sx={{ p: 3 }}>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
                  <Typography variant="h6" sx={{ fontWeight: 700 }}>
                    {storeId}
                  </Typography>
                  <Chip
                    icon={getWeatherIcon(weather.weather_condition)}
                    label={weather.weather_condition}
                    color={getWeatherColor(weather.weather_condition)}
                    size="medium"
                  />
                </Box>

                <Grid container spacing={3} sx={{ mb: 3 }}>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      Temperature
                    </Typography>
                    <Typography variant="h5" color="primary" sx={{ fontWeight: 700 }}>
                      {weather.temperature}°C
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      Humidity
                    </Typography>
                    <Typography variant="h5" color="secondary" sx={{ fontWeight: 700 }}>
                      {weather.humidity}%
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      Wind Speed
                    </Typography>
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      {weather.wind_speed} km/h
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      Visibility
                    </Typography>
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      {weather.visibility} km
                    </Typography>
                  </Grid>
                </Grid>

                <Box sx={{ mb: 3 }}>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                    Weather Impact
                  </Typography>
                  <Chip
                    label={weather.weather_condition === 'rain' ? 'High' : 'Normal'}
                    color={weather.weather_condition === 'rain' ? 'warning' : 'success'}
                    size="medium"
                  />
                </Box>

                {/* Holiday Impact for selected SKUs only */}
                <Box>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    <Event sx={{ mr: 0.5, fontSize: 18, verticalAlign: 'middle' }} />
                    Holiday Impact (Christmas)
                  </Typography>
                  <Grid container spacing={2}>
                    {displaySKUs.map((skuId) => {
                      const impact = getHolidayImpact(skuId);
                      return (
                        <Grid item xs={12} key={skuId}>
                          <Box sx={{ 
                            p: 2, 
                            borderRadius: 2, 
                            background: 'rgba(255, 255, 255, 0.05)',
                            border: '1px solid #333333'
                          }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <Box>
                                <Typography variant="body1" sx={{ fontWeight: 600, mb: 0.5 }}>
                                  {getSKUName(skuId)}
                                </Typography>
                                <Typography variant="caption" sx={{ opacity: 0.7 }}>
                                  {skuId}
                                </Typography>
                              </Box>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <TrendingUp sx={{ fontSize: 18, color: '#ffc220' }} />
                                <Typography variant="h6" sx={{ fontWeight: 700 }}>
                                  {impact}x
                                </Typography>
                              </Box>
                            </Box>
                          </Box>
                        </Grid>
                      );
                    })}
                  </Grid>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}

        {/* Holiday Calendar - Wider Layout */}
        <Grid item xs={12}>
          <Card>
            <CardContent sx={{ p: 4 }}>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 700, mb: 3 }}>
                Upcoming Events & Holidays
              </Typography>
              
              <Grid container spacing={3}>
                {holidays.map((holiday) => (
                  <Grid item xs={12} md={4} key={holiday.name}>
                    <Paper sx={{ p: 3, bgcolor: `${holiday.color}20`, border: `1px solid ${holiday.color}40`, height: '100%' }}>
                      <Typography variant="h6" fontWeight="bold" sx={{ color: holiday.color, mb: 1 }}>
                        {holiday.name}
                      </Typography>
                      <Typography variant="body2" sx={{ opacity: 0.8, mb: 2 }}>
                        {holiday.date}
                      </Typography>
                      <Box>
                        <Typography variant="body2" sx={{ opacity: 0.7, mb: 2, fontWeight: 600 }}>
                          Selected Products Impact:
                        </Typography>
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                          {displaySKUs.map((skuId) => {
                            const impact = holiday.impact[skuId] || 1.0;
                            return (
                              <Box key={skuId} sx={{ 
                                display: 'flex', 
                                justifyContent: 'space-between', 
                                alignItems: 'center',
                                p: 1.5,
                                borderRadius: 1,
                                background: 'rgba(255, 255, 255, 0.1)'
                              }}>
                                <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                  {getSKUName(skuId)}
                                </Typography>
                                <Chip
                                  label={`${impact}x`}
                                  size="small"
                                  sx={{ 
                                    background: holiday.color,
                                    color: 'white',
                                    fontWeight: 600
                                  }}
                                />
                              </Box>
                            );
                          })}
                        </Box>
                      </Box>
                    </Paper>
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default WeatherWidget; 