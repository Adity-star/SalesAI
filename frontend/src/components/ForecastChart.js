import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  AreaChart,
  ComposedChart,
  Bar,
  BarChart
} from 'recharts';
import {
  Box,
  Paper,
  Typography,
  Card,
  CardContent,
  Grid,
  Chip,
  CircularProgress,
  Alert,
  Divider,
  Fade,
  Zoom
} from '@mui/material';
import { 
  TrendingUp, 
  Assessment, 
  Timeline, 
  Analytics,
  ShowChart,
  BarChart as BarChartIcon
} from '@mui/icons-material';

const ForecastChart = ({ data, loading }) => {
  if (loading) {
    return (
      <Box sx={{ 
        display: 'flex', 
        flexDirection: 'column',
        justifyContent: 'center', 
        alignItems: 'center', 
        minHeight: '400px',
        gap: 2
      }}>
        <CircularProgress size={60} sx={{ color: '#0071ce' }} />
        <Typography variant="h6" sx={{ fontWeight: 600, color: '#0071ce' }}>
          Generating Forecast...
        </Typography>
        <Typography variant="body2" sx={{ opacity: 0.7 }}>
          Please wait while our AI models process your data
        </Typography>
      </Box>
    );
  }

  if (!data || !data.forecasts) {
    return (
      <Fade in timeout={800}>
        <Paper sx={{ 
          p: 6, 
          textAlign: 'center',
          background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
          border: '1px solid #333333',
          borderRadius: 4
        }}>
          <Analytics sx={{ 
            fontSize: 80, 
            color: '#0071ce', 
            mb: 3,
            opacity: 0.6
          }} />
          <Typography variant="h4" sx={{ 
            fontWeight: 700, 
            mb: 2,
            background: 'linear-gradient(45deg, #0071ce 30%, #0056a3 90%)',
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent'
          }}>
            No Forecast Data Available
          </Typography>
          <Typography variant="body1" sx={{ 
            opacity: 0.7,
            maxWidth: 400,
            mx: 'auto',
            lineHeight: 1.6
          }}>
            Generate a forecast to visualize demand predictions and analytics insights
          </Typography>
        </Paper>
      </Fade>
    );
  }

  // Transform forecast data for charts
  const transformForecastData = () => {
    const chartData = [];
    const forecastDays = data.forecast_days;
    
    // Generate dates for x-axis
    const dates = [];
    const today = new Date();
    for (let i = 1; i <= forecastDays; i++) {
      const date = new Date(today);
      date.setDate(today.getDate() + i);
      dates.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
    }

    // Combine all forecasts
    Object.entries(data.forecasts).forEach(([key, forecast]) => {
      const { sku_id, store_id, forecast: values, confidence_interval } = forecast;
      
      values.forEach((value, index) => {
        const existingData = chartData.find(d => d.date === dates[index]);
        
        if (existingData) {
          existingData[`${sku_id}_${store_id}`] = Math.round(value);
          if (confidence_interval && confidence_interval.lower) {
            existingData[`${sku_id}_${store_id}_lower`] = Math.round(confidence_interval.lower[index] || value * 0.8);
            existingData[`${sku_id}_${store_id}_upper`] = Math.round(confidence_interval.upper[index] || value * 1.2);
          }
        } else {
          const newData = {
            date: dates[index],
            [`${sku_id}_${store_id}`]: Math.round(value)
          };
          if (confidence_interval && confidence_interval.lower) {
            newData[`${sku_id}_${store_id}_lower`] = Math.round(confidence_interval.lower[index] || value * 0.8);
            newData[`${sku_id}_${store_id}_upper`] = Math.round(confidence_interval.upper[index] || value * 1.2);
          }
          chartData.push(newData);
        }
      });
    });

    return chartData.sort((a, b) => new Date(a.date) - new Date(b.date));
  };

  const chartData = transformForecastData();
  
  // Enhanced color palette
  const colors = [
    '#0071ce', // Walmart blue
    '#ffc220', // Walmart yellow
    '#4caf50', // Green
    '#ff9800', // Orange
    '#9c27b0', // Purple
    '#f44336', // Red
    '#00bcd4', // Cyan
    '#795548'  // Brown
  ];
  
  const getForecastKeys = () => {
    return Object.keys(data.forecasts).map(key => {
      const forecast = data.forecasts[key];
      return `${forecast.sku_id}_${forecast.store_id}`;
    });
  };

  const forecastKeys = getForecastKeys();

  // Calculate summary statistics
  const calculateSummary = () => {
    const summary = {};
    
    Object.entries(data.forecasts).forEach(([key, forecast]) => {
      const { sku_id, store_id, forecast: values } = forecast;
      const total = values.reduce((sum, val) => sum + val, 0);
      const avg = total / values.length;
      const max = Math.max(...values);
      const min = Math.min(...values);
      const trend = values[values.length - 1] > values[0] ? 'up' : 'down';
      
      summary[key] = {
        sku_id,
        store_id,
        total: Math.round(total),
        average: Math.round(avg),
        max: Math.round(max),
        min: Math.round(min),
        trend
      };
    });
    
    return summary;
  };

  const summary = calculateSummary();

  // Custom tooltip component
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <Box sx={{
          background: 'rgba(26, 26, 26, 0.95)',
          border: '1px solid #333333',
          borderRadius: 2,
          p: 2,
          backdropFilter: 'blur(10px)'
        }}>
          <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
            {label}
          </Typography>
          {payload.map((entry, index) => (
            <Typography key={index} variant="body2" sx={{ color: entry.color }}>
              {entry.name}: {Math.round(entry.value)}
            </Typography>
          ))}
        </Box>
      );
    }
    return null;
  };

  return (
    <Fade in timeout={1000}>
      <Box>
        {/* Header */}
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 4 }}>
          <Box sx={{ 
            p: 2, 
            borderRadius: 3, 
            background: 'linear-gradient(45deg, #4caf50 30%, #66bb6a 90%)',
            mr: 3,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <ShowChart sx={{ color: 'white', fontSize: 28 }} />
          </Box>
          <Box>
            <Typography variant="h4" sx={{ 
              fontWeight: 700,
              mb: 0.5,
              letterSpacing: '-0.01em'
            }}>
              Demand Forecast Analytics
            </Typography>
            <Typography variant="body1" sx={{ 
              opacity: 0.7,
              fontWeight: 500
            }}>
              AI-powered demand predictions with confidence intervals
            </Typography>
          </Box>
        </Box>
        
        <Grid container spacing={4}>
          {/* Main Forecast Chart */}
          <Grid item xs={12}>
            <Card sx={{ 
              background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
              border: '1px solid #333333',
              borderRadius: 4,
              overflow: 'hidden'
            }}>
              <CardContent sx={{ p: 4 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                  <TrendingUp sx={{ mr: 2, color: '#0071ce', fontSize: 28 }} />
                  <Typography variant="h5" sx={{ fontWeight: 700 }}>
                    Forecast Trends
                  </Typography>
                  <Chip 
                    label={`${data.model_used.toUpperCase()} Model`}
                    sx={{ 
                      ml: 2,
                      background: 'linear-gradient(45deg, #0071ce 30%, #0056a3 90%)',
                      color: 'white',
                      fontWeight: 600
                    }}
                  />
                </Box>
                
                <ResponsiveContainer width="100%" height={450}>
                  <ComposedChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333333" opacity={0.3} />
                    <XAxis 
                      dataKey="date" 
                      stroke="#b0b0b0"
                      fontSize={12}
                      fontWeight={500}
                    />
                    <YAxis 
                      stroke="#b0b0b0"
                      fontSize={12}
                      fontWeight={500}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend 
                      wrapperStyle={{
                        paddingTop: 20,
                        fontSize: 12,
                        fontWeight: 600
                      }}
                    />
                    {forecastKeys.map((key, index) => (
                      <Line
                        key={key}
                        type="monotone"
                        dataKey={key}
                        stroke={colors[index % colors.length]}
                        strokeWidth={3}
                        dot={{ r: 4, strokeWidth: 2, fill: colors[index % colors.length] }}
                        activeDot={{ r: 6, strokeWidth: 2 }}
                        name={`${key.split('_')[0]} - ${key.split('_')[1]}`}
                      />
                    ))}
                  </ComposedChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </Grid>

          {/* Summary Statistics */}
          <Grid item xs={12}>
            <Typography variant="h5" sx={{ 
              fontWeight: 700, 
              mb: 3,
              display: 'flex',
              alignItems: 'center'
            }}>
              <Assessment sx={{ mr: 2, color: '#ffc220' }} />
              Forecast Summary
            </Typography>
            
            <Grid container spacing={3}>
              {Object.entries(summary).map(([key, data], index) => (
                <Grid item xs={12} sm={6} md={4} key={key}>
                  <Zoom in timeout={800 + index * 200}>
                    <Card sx={{ 
                      height: '100%',
                      background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
                      border: '1px solid #333333',
                      transition: 'all 0.3s ease-in-out',
                      '&:hover': {
                        transform: 'translateY(-4px)',
                        boxShadow: '0 12px 40px rgba(0, 0, 0, 0.4)',
                        border: '1px solid #0071ce'
                      }
                    }}>
                      <CardContent sx={{ p: 3 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                          <Box sx={{ 
                            p: 1, 
                            borderRadius: 2, 
                            background: `linear-gradient(45deg, ${colors[index % colors.length]} 30%, ${colors[(index + 1) % colors.length]} 90%)`,
                            mr: 2
                          }}>
                            <Analytics sx={{ color: 'white', fontSize: 20 }} />
                          </Box>
                          <Typography variant="h6" sx={{ fontWeight: 700 }}>
                            {data.sku_id} - {data.store_id}
                          </Typography>
                        </Box>
                        
                        <Divider sx={{ mb: 2, borderColor: '#333333' }} />
                        
                        <Grid container spacing={2}>
                          <Grid item xs={6}>
                            <Box sx={{ textAlign: 'center' }}>
                              <Typography variant="h4" sx={{ 
                                fontWeight: 800,
                                color: '#0071ce',
                                mb: 0.5
                              }}>
                                {data.total.toLocaleString()}
                              </Typography>
                              <Typography variant="caption" sx={{ 
                                opacity: 0.7,
                                fontWeight: 600,
                                textTransform: 'uppercase',
                                letterSpacing: '0.05em'
                              }}>
                                Total Units
                              </Typography>
                            </Box>
                          </Grid>
                          
                          <Grid item xs={6}>
                            <Box sx={{ textAlign: 'center' }}>
                              <Typography variant="h4" sx={{ 
                                fontWeight: 800,
                                color: '#4caf50',
                                mb: 0.5
                              }}>
                                {data.average.toLocaleString()}
                              </Typography>
                              <Typography variant="caption" sx={{ 
                                opacity: 0.7,
                                fontWeight: 600,
                                textTransform: 'uppercase',
                                letterSpacing: '0.05em'
                              }}>
                                Daily Avg
                              </Typography>
                            </Box>
                          </Grid>
                          
                          <Grid item xs={6}>
                            <Box sx={{ textAlign: 'center' }}>
                              <Typography variant="h6" sx={{ 
                                fontWeight: 700,
                                color: '#ff9800',
                                mb: 0.5
                              }}>
                                {data.max.toLocaleString()}
                              </Typography>
                              <Typography variant="caption" sx={{ 
                                opacity: 0.7,
                                fontWeight: 600,
                                textTransform: 'uppercase',
                                letterSpacing: '0.05em'
                              }}>
                                Peak Demand
                              </Typography>
                            </Box>
                          </Grid>
                          
                          <Grid item xs={6}>
                            <Box sx={{ textAlign: 'center' }}>
                              <Typography variant="h6" sx={{ 
                                fontWeight: 700,
                                color: '#f44336',
                                mb: 0.5
                              }}>
                                {data.min.toLocaleString()}
                              </Typography>
                              <Typography variant="caption" sx={{ 
                                opacity: 0.7,
                                fontWeight: 600,
                                textTransform: 'uppercase',
                                letterSpacing: '0.05em'
                              }}>
                                Min Demand
                              </Typography>
                            </Box>
                          </Grid>
                        </Grid>
                        
                        <Box sx={{ 
                          mt: 2, 
                          p: 1.5, 
                          borderRadius: 2,
                          background: data.trend === 'up' 
                            ? 'linear-gradient(45deg, #4caf50 20%, #66bb6a 80%)'
                            : 'linear-gradient(45deg, #f44336 20%, #e57373 80%)',
                          textAlign: 'center'
                        }}>
                          <Typography variant="body2" sx={{ 
                            fontWeight: 700,
                            color: 'white',
                            textTransform: 'uppercase',
                            letterSpacing: '0.05em'
                          }}>
                            Trend: {data.trend === 'up' ? '↗ Increasing' : '↘ Decreasing'}
                          </Typography>
                        </Box>
                      </CardContent>
                    </Card>
                  </Zoom>
                </Grid>
              ))}
            </Grid>
          </Grid>

          {/* Model Information */}
          <Grid item xs={12}>
            <Card sx={{ 
              background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
              border: '1px solid #333333',
              borderRadius: 4
            }}>
              <CardContent sx={{ p: 4 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                  <Timeline sx={{ mr: 2, color: '#ffc220', fontSize: 28 }} />
                  <Typography variant="h5" sx={{ fontWeight: 700 }}>
                    Model Information
                  </Typography>
                </Box>
                
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <Box sx={{ 
                      p: 3, 
                      borderRadius: 3,
                      background: 'linear-gradient(45deg, #0071ce 20%, #0056a3 80%)',
                      color: 'white'
                    }}>
                      <Typography variant="h6" sx={{ fontWeight: 700, mb: 1 }}>
                        Model Type
                      </Typography>
                      <Typography variant="h4" sx={{ fontWeight: 800, mb: 2 }}>
                        {data.model_used.toUpperCase()}
                      </Typography>
                      <Typography variant="body2" sx={{ opacity: 0.9 }}>
                        Advanced machine learning model for demand forecasting
                      </Typography>
                    </Box>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Box sx={{ 
                      p: 3, 
                      borderRadius: 3,
                      background: 'linear-gradient(45deg, #4caf50 20%, #66bb6a 80%)',
                      color: 'white'
                    }}>
                      <Typography variant="h6" sx={{ fontWeight: 700, mb: 1 }}>
                        Forecast Period
                      </Typography>
                      <Typography variant="h4" sx={{ fontWeight: 800, mb: 2 }}>
                        {data.forecast_days} Days
                      </Typography>
                      <Typography variant="body2" sx={{ opacity: 0.9 }}>
                        Long-term demand prediction with confidence intervals
                      </Typography>
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>
    </Fade>
  );
};

export default ForecastChart; 