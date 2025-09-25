import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Card,
  CardContent,
  Grid,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Alert,
  Chip,
  Divider,
  Fade,
  Zoom
} from '@mui/material';
import {
  Science,
  TrendingUp,
  TrendingDown,
  CompareArrows,
  Analytics,
  PlayArrow
} from '@mui/icons-material';
import { runSimulation } from '../services/api';

const SimulationPanel = ({ forecastData }) => {
  const [scenario, setScenario] = useState({
    weather_impact: 1.2,
    promotion_impact: 1.5,
    holiday_impact: 1.3,
    price_change: -10,
    competitor_action: 'none'
  });
  const [simulationResult, setSimulationResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Holiday data for simulation
  const holidays = [
    { name: 'Christmas', impact: 2.5, date: 'Dec 25, 2024' },
    { name: 'New Year\'s Eve', impact: 1.7, date: 'Dec 31, 2024' },
    { name: 'Valentine\'s Day', impact: 1.8, date: 'Feb 14, 2025' },
    { name: 'Thanksgiving', impact: 2.2, date: 'Nov 28, 2024' },
    { name: 'Black Friday', impact: 3.0, date: 'Nov 29, 2024' }
  ];

  const competitorActions = [
    { value: 'none', label: 'No Action', impact: 1.0 },
    { value: 'price_cut', label: 'Price Cut', impact: 0.8 },
    { value: 'promotion', label: 'Promotion', impact: 0.9 },
    { value: 'new_product', label: 'New Product', impact: 0.7 },
    { value: 'aggressive', label: 'Aggressive Marketing', impact: 0.6 }
  ];

  const runSimulationScenario = async () => {
    if (!forecastData || !forecastData.forecasts) {
      // Generate sample forecast data for testing
      const sampleForecastData = {
        forecasts: {
          'SKU001_STORE001': {
            sku_id: 'SKU001',
            store_id: 'STORE001',
            forecast: [45, 48, 52, 49, 51, 47, 53, 50, 55, 48, 52, 49, 51, 54, 47, 50, 53, 49, 52, 48, 51, 55, 49, 53, 50, 52, 47, 51, 54, 48],
            confidence_interval: {
              lower: [36, 38, 42, 39, 41, 38, 42, 40, 44, 38, 42, 39, 41, 43, 38, 40, 42, 39, 42, 38, 41, 44, 39, 42, 40, 42, 38, 41, 43, 38],
              upper: [54, 58, 62, 59, 61, 56, 64, 60, 66, 58, 62, 59, 61, 65, 56, 60, 64, 59, 62, 58, 61, 66, 59, 64, 60, 62, 56, 61, 65, 58]
            }
          },
          'SKU002_STORE001': {
            sku_id: 'SKU002',
            store_id: 'STORE001',
            forecast: [120, 125, 118, 122, 127, 124, 129, 121, 126, 123, 128, 125, 130, 122, 127, 124, 129, 126, 131, 123, 128, 125, 130, 127, 132, 124, 129, 126, 131, 128],
            confidence_interval: {
              lower: [96, 100, 94, 98, 102, 99, 103, 97, 101, 98, 102, 100, 104, 98, 102, 99, 103, 101, 105, 98, 102, 100, 104, 102, 106, 99, 103, 101, 105, 102],
              upper: [144, 150, 142, 146, 152, 149, 155, 145, 151, 148, 154, 150, 156, 146, 152, 149, 155, 151, 157, 148, 154, 150, 156, 152, 158, 149, 155, 151, 157, 154]
            }
          }
        },
        model_used: 'arima',
        forecast_days: 30
      };
      
      setError('Using sample data for simulation. Generate a real forecast for production use.');
    }

    setLoading(true);
    setError(null);

    try {
      // Use either real forecast data or sample data
      const dataToUse = forecastData || {
        forecasts: {
          'SKU001_STORE001': {
            sku_id: 'SKU001',
            store_id: 'STORE001',
            forecast: [45, 48, 52, 49, 51, 47, 53, 50, 55, 48, 52, 49, 51, 54, 47, 50, 53, 49, 52, 48, 51, 55, 49, 53, 50, 52, 47, 51, 54, 48],
            confidence_interval: {
              lower: [36, 38, 42, 39, 41, 38, 42, 40, 44, 38, 42, 39, 41, 43, 38, 40, 42, 39, 42, 38, 41, 44, 39, 42, 40, 42, 38, 41, 43, 38],
              upper: [54, 58, 62, 59, 61, 56, 64, 60, 66, 58, 62, 59, 61, 65, 56, 60, 64, 59, 62, 58, 61, 66, 59, 64, 60, 62, 56, 61, 65, 58]
            }
          },
          'SKU002_STORE001': {
            sku_id: 'SKU002',
            store_id: 'STORE001',
            forecast: [120, 125, 118, 122, 127, 124, 129, 121, 126, 123, 128, 125, 130, 122, 127, 124, 129, 126, 131, 123, 128, 125, 130, 127, 132, 124, 129, 126, 131, 128],
            confidence_interval: {
              lower: [96, 100, 94, 98, 102, 99, 103, 97, 101, 98, 102, 100, 104, 98, 102, 99, 103, 101, 105, 98, 102, 100, 104, 102, 106, 99, 103, 101, 105, 102],
              upper: [144, 150, 142, 146, 152, 149, 155, 145, 151, 148, 154, 150, 156, 146, 152, 149, 155, 151, 157, 148, 154, 150, 156, 152, 158, 149, 155, 151, 157, 154]
            }
          }
        },
        model_used: 'arima',
        forecast_days: 30
      };

      // Create modified forecast based on scenario
      const modifiedForecast = JSON.parse(JSON.stringify(dataToUse));
      
      Object.keys(modifiedForecast.forecasts).forEach(key => {
        const forecast = modifiedForecast.forecasts[key];
        
        // Apply scenario impacts
        const weatherMultiplier = scenario.weather_impact;
        const promotionMultiplier = scenario.promotion_impact;
        const holidayMultiplier = scenario.holiday_impact;
        
        // Fix price multiplier calculation - price increase should decrease demand
        const priceMultiplier = scenario.price_change === 0 
          ? 1.0  // No price change
          : scenario.price_change > 0 
            ? 1 / (1 + (scenario.price_change / 100))  // Price increase reduces demand
            : 1 + (Math.abs(scenario.price_change) / 100);  // Price decrease increases demand
        
        const competitorMultiplier = competitorActions.find(c => c.value === scenario.competitor_action)?.impact || 1.0;
        
        // Calculate total impact
        const totalMultiplier = weatherMultiplier * promotionMultiplier * holidayMultiplier * priceMultiplier * competitorMultiplier;
        
        console.log(`Simulation for ${key}:`, {
          weather: weatherMultiplier,
          promotion: promotionMultiplier,
          holiday: holidayMultiplier,
          price: priceMultiplier,
          competitor: competitorMultiplier,
          total: totalMultiplier,
          beforeValues: forecast.forecast.slice(0, 3) // First 3 values before change
        });
        
        // Apply to forecast values
        forecast.forecast = forecast.forecast.map(value => Math.max(0, value * totalMultiplier));
        
        console.log(`After change for ${key}:`, {
          afterValues: forecast.forecast.slice(0, 3) // First 3 values after change
        });
        
        // Update confidence intervals
        if (forecast.confidence_interval) {
          forecast.confidence_interval.lower = forecast.confidence_interval.lower.map(v => Math.max(0, v * totalMultiplier));
          forecast.confidence_interval.upper = forecast.confidence_interval.upper.map(v => Math.max(0, v * totalMultiplier));
        }
      });

      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Log to verify data separation
      console.log('Original data sample:', {
        key: Object.keys(dataToUse.forecasts)[0],
        originalValues: dataToUse.forecasts[Object.keys(dataToUse.forecasts)[0]].forecast.slice(0, 3)
      });
      
      console.log('Modified data sample:', {
        key: Object.keys(modifiedForecast.forecasts)[0],
        modifiedValues: modifiedForecast.forecasts[Object.keys(modifiedForecast.forecasts)[0]].forecast.slice(0, 3)
      });

      setSimulationResult({
        original: dataToUse,
        modified: modifiedForecast,
        scenario: scenario,
        summary: calculateSummary(dataToUse, modifiedForecast)
      });

    } catch (err) {
      setError('Failed to run simulation. Please try again.');
      console.error('Simulation error:', err);
    } finally {
      setLoading(false);
    }
  };

  const calculateSummary = (original, modified) => {
    const summary = {};
    
    Object.keys(original.forecasts).forEach(key => {
      const originalForecast = original.forecasts[key];
      const modifiedForecast = modified.forecasts[key];
      
      const originalTotal = originalForecast.forecast.reduce((sum, val) => sum + val, 0);
      const modifiedTotal = modifiedForecast.forecast.reduce((sum, val) => sum + val, 0);
      
      // Fix percentage calculation to handle edge cases
      let change = 0;
      if (originalTotal > 0) {
        change = ((modifiedTotal - originalTotal) / originalTotal) * 100;
      } else if (modifiedTotal > 0) {
        change = 100; // If original was 0 and modified is > 0, that's a 100% increase
      }
      
      console.log(`Summary for ${key}:`, {
        originalTotal: originalTotal,
        modifiedTotal: modifiedTotal,
        change: change,
        originalValues: originalForecast.forecast.slice(0, 5), // First 5 values
        modifiedValues: modifiedForecast.forecast.slice(0, 5)  // First 5 values
      });
      
      summary[key] = {
        originalTotal: Math.round(originalTotal),
        modifiedTotal: Math.round(modifiedTotal),
        change: Math.round(change * 100) / 100, // Round to 2 decimal places
        changeType: change > 0 ? 'increase' : change < 0 ? 'decrease' : 'no change'
      };
    });
    
    return summary;
  };

  const resetScenario = () => {
    setScenario({
      weather_impact: 1.0,
      promotion_impact: 1.0,
      holiday_impact: 1.0,
      price_change: 0,
      competitor_action: 'none'
    });
    setSimulationResult(null);
    setError(null);
  };

  if (!forecastData || !forecastData.forecasts) {
    return (
      <Fade in timeout={800}>
        <Paper sx={{ 
          p: 6, 
          textAlign: 'center',
          background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
          border: '1px solid #333333',
          borderRadius: 4
        }}>
          <Science sx={{ 
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
            Ready for Simulation
          </Typography>
          <Typography variant="body1" sx={{ 
            opacity: 0.7,
            maxWidth: 400,
            mx: 'auto',
            lineHeight: 1.6,
            mb: 3
          }}>
            No forecast data available. You can run simulation with sample data or generate a real forecast first.
          </Typography>
          <Button
            variant="contained"
            onClick={runSimulationScenario}
            sx={{ 
              background: 'linear-gradient(45deg, #0071ce 30%, #0056a3 90%)',
              fontWeight: 600
            }}
          >
            Run Simulation with Sample Data
          </Button>
        </Paper>
      </Fade>
    );
  }

  return (
    <Fade in timeout={1000}>
      <Box>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 4 }}>
          <Box sx={{ 
            p: 2, 
            borderRadius: 3, 
            background: 'linear-gradient(45deg, #ffc220 30%, #ff8f00 90%)',
            mr: 3,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <Science sx={{ color: 'white', fontSize: 28 }} />
          </Box>
          <Box>
            <Typography variant="h4" sx={{ 
              fontWeight: 700,
              mb: 0.5,
              letterSpacing: '-0.01em'
            }}>
              What-If Simulation Mode
            </Typography>
            <Typography variant="body1" sx={{ 
              opacity: 0.7,
              fontWeight: 500
            }}>
              Test different scenarios and their impact on demand forecasts
            </Typography>
          </Box>
        </Box>

        <Grid container spacing={4}>
          {/* Scenario Configuration */}
          <Grid item xs={12} md={4}>
            <Card sx={{ 
              height: 'fit-content',
              background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
              border: '1px solid #333333',
              borderRadius: 4
            }}>
              <CardContent sx={{ p: 3 }}>
                <Typography variant="h6" sx={{ fontWeight: 700, mb: 3 }}>
                  Scenario Parameters
                </Typography>
                
                {/* Weather Impact */}
                <Box sx={{ mb: 3 }}>
                  <Typography gutterBottom sx={{ fontWeight: 600 }}>
                    Weather Impact
                  </Typography>
                  <Slider
                    value={scenario.weather_impact}
                    onChange={(e, value) => setScenario({...scenario, weather_impact: value})}
                    min={0.5}
                    max={2.0}
                    step={0.1}
                    marks={[
                      {value: 0.5, label: '0.5x'},
                      {value: 1.0, label: '1.0x'},
                      {value: 2.0, label: '2.0x'}
                    ]}
                    sx={{ color: '#0071ce' }}
                  />
                  <Typography variant="body2" sx={{ opacity: 0.7 }}>
                    {scenario.weather_impact}x demand impact
                  </Typography>
                </Box>

                {/* Promotion Impact */}
                <Box sx={{ mb: 3 }}>
                  <Typography gutterBottom sx={{ fontWeight: 600 }}>
                    Promotion Impact
                  </Typography>
                  <Slider
                    value={scenario.promotion_impact}
                    onChange={(e, value) => setScenario({...scenario, promotion_impact: value})}
                    min={0.5}
                    max={3.0}
                    step={0.1}
                    marks={[
                      {value: 0.5, label: '0.5x'},
                      {value: 1.0, label: '1.0x'},
                      {value: 3.0, label: '3.0x'}
                    ]}
                    sx={{ color: '#ffc220' }}
                  />
                  <Typography variant="body2" sx={{ opacity: 0.7 }}>
                    {scenario.promotion_impact}x demand impact
                  </Typography>
                </Box>

                {/* Holiday Impact */}
                <Box sx={{ mb: 3 }}>
                  <Typography gutterBottom sx={{ fontWeight: 600 }}>
                    Holiday Impact
                  </Typography>
                  <Slider
                    value={scenario.holiday_impact}
                    onChange={(e, value) => setScenario({...scenario, holiday_impact: value})}
                    min={0.5}
                    max={2.5}
                    step={0.1}
                    marks={[
                      {value: 0.5, label: '0.5x'},
                      {value: 1.0, label: '1.0x'},
                      {value: 2.5, label: '2.5x'}
                    ]}
                    sx={{ color: '#e74c3c' }}
                  />
                  <Typography variant="body2" sx={{ opacity: 0.7 }}>
                    {scenario.holiday_impact}x demand impact
                  </Typography>
                </Box>

                {/* Price Change */}
                <Box sx={{ mb: 3 }}>
                  <Typography gutterBottom sx={{ fontWeight: 600 }}>
                    Price Change (%)
                  </Typography>
                  <TextField
                    type="number"
                    value={scenario.price_change}
                    onChange={(e) => setScenario({...scenario, price_change: parseFloat(e.target.value) || 0})}
                    fullWidth
                    size="small"
                    inputProps={{ min: -50, max: 50, step: 1 }}
                    sx={{ 
                      '& .MuiOutlinedInput-root': {
                        '& fieldset': { borderColor: '#333333' },
                        '&:hover fieldset': { borderColor: '#0071ce' },
                        '&.Mui-focused fieldset': { borderColor: '#0071ce' }
                      }
                    }}
                  />
                </Box>

                {/* Competitor Action */}
                <Box sx={{ mb: 3 }}>
                  <FormControl fullWidth size="small">
                    <InputLabel sx={{ color: '#b0b0b0' }}>Competitor Action</InputLabel>
                    <Select
                      value={scenario.competitor_action}
                      onChange={(e) => setScenario({...scenario, competitor_action: e.target.value})}
                      sx={{ 
                        '& .MuiOutlinedInput-notchedOutline': { borderColor: '#333333' },
                        '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: '#0071ce' },
                        '&.Mui-focused .MuiOutlinedInput-notchedOutline': { borderColor: '#0071ce' }
                      }}
                    >
                      {competitorActions.map((action) => (
                        <MenuItem key={action.value} value={action.value}>
                          {action.label}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Box>

                {/* Action Buttons */}
                <Box sx={{ display: 'flex', gap: 2 }}>
                  <Button
                    variant="contained"
                    onClick={runSimulationScenario}
                    disabled={loading}
                    fullWidth
                    sx={{ 
                      background: 'linear-gradient(45deg, #0071ce 30%, #0056a3 90%)',
                      fontWeight: 600
                    }}
                  >
                    {loading ? (
                      'Running...'
                    ) : (
                      <>
                        <PlayArrow sx={{ mr: 1 }} />
                        Run Simulation
                      </>
                    )}
                  </Button>
                  <Button
                    variant="outlined"
                    onClick={resetScenario}
                    disabled={loading}
                    sx={{ 
                      borderColor: '#ffc220',
                      color: '#ffc220',
                      '&:hover': { borderColor: '#ff8f00' }
                    }}
                  >
                    Reset
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Simulation Results */}
          <Grid item xs={12} md={8}>
            {error && (
              <Alert severity="error" sx={{ mb: 3, borderRadius: 2 }}>
                {error}
              </Alert>
            )}

            {simulationResult ? (
              <Zoom in timeout={800}>
                <Card sx={{ 
                  background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
                  border: '1px solid #333333',
                  borderRadius: 4
                }}>
                  <CardContent sx={{ p: 4 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                      <Analytics sx={{ mr: 2, color: '#0071ce', fontSize: 28 }} />
                      <Typography variant="h5" sx={{ fontWeight: 700 }}>
                        Simulation Results
                      </Typography>
                    </Box>
                    
                    <Alert severity="info" sx={{ mb: 3, borderRadius: 2 }}>
                      <Typography variant="body2">
                        <strong>Scenario Applied:</strong> Weather: {scenario.weather_impact}x, 
                        Promotion: {scenario.promotion_impact}x, 
                        Holiday: {scenario.holiday_impact}x,
                        Price: {scenario.price_change > 0 ? '+' : ''}{scenario.price_change}%,
                        Competitor: {competitorActions.find(c => c.value === scenario.competitor_action)?.label}
                      </Typography>
                    </Alert>

                    {/* Debug Information */}
                    <Alert severity="warning" sx={{ mb: 3, borderRadius: 2 }}>
                      <Typography variant="body2">
                        <strong>Debug Info:</strong> Total multiplier: {
                          (scenario.weather_impact * 
                           scenario.promotion_impact * 
                           scenario.holiday_impact * 
                           (scenario.price_change === 0 ? 1.0 : scenario.price_change > 0 ? 1 / (1 + (scenario.price_change / 100)) : 1 + (Math.abs(scenario.price_change) / 100)) * 
                           (competitorActions.find(c => c.value === scenario.competitor_action)?.impact || 1.0)).toFixed(3)
                        }x
                      </Typography>
                      <Typography variant="body2" sx={{ mt: 1 }}>
                        <strong>Expected Change:</strong> {
                          ((scenario.weather_impact * 
                           scenario.promotion_impact * 
                           scenario.holiday_impact * 
                           (scenario.price_change === 0 ? 1.0 : scenario.price_change > 0 ? 1 / (1 + (scenario.price_change / 100)) : 1 + (Math.abs(scenario.price_change) / 100)) * 
                           (competitorActions.find(c => c.value === scenario.competitor_action)?.impact || 1.0)) - 1) * 100
                        }%
                      </Typography>
                    </Alert>

                    <Grid container spacing={3}>
                      {Object.entries(simulationResult.summary).map(([key, data]) => {
                        console.log(`Rendering result for ${key}:`, data);
                        return (
                          <Grid item xs={12} sm={6} key={key}>
                            <Card sx={{ 
                              background: 'rgba(255, 255, 255, 0.05)',
                              border: '1px solid #333333',
                              borderRadius: 3
                            }}>
                              <CardContent sx={{ p: 3 }}>
                                <Typography variant="h6" sx={{ fontWeight: 700, mb: 2 }}>
                                  {key}
                                </Typography>
                                
                                {/* Debug info for this specific result */}
                                <Box sx={{ mb: 2, p: 1, background: 'rgba(255, 255, 0, 0.1)', borderRadius: 1 }}>
                                  <Typography variant="caption" sx={{ opacity: 0.7 }}>
                                    Debug: Change = {data.change}%, Type = {data.changeType}
                                  </Typography>
                                </Box>
                              
                              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                                <Typography variant="body2" sx={{ opacity: 0.7 }}>
                                  Original
                                </Typography>
                                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                                  {data.originalTotal.toLocaleString()}
                                </Typography>
                              </Box>
                              
                              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                                <Typography variant="body2" sx={{ opacity: 0.7 }}>
                                  Modified
                                </Typography>
                                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                                  {data.modifiedTotal.toLocaleString()}
                                </Typography>
                              </Box>
                              
                              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                                <Typography variant="body2" sx={{ opacity: 0.7 }}>
                                  Difference
                                </Typography>
                                <Typography variant="h6" sx={{ 
                                  fontWeight: 600,
                                  color: data.change > 0 ? '#4caf50' : data.change < 0 ? '#f44336' : 'inherit'
                                }}>
                                  {data.change > 0 ? '+' : ''}{(data.modifiedTotal - data.originalTotal).toLocaleString()}
                                </Typography>
                              </Box>
                              
                              <Divider sx={{ my: 2, borderColor: '#333333' }} />
                              
                              <Box sx={{ 
                                display: 'flex', 
                                alignItems: 'center', 
                                justifyContent: 'space-between',
                                p: 2,
                                borderRadius: 2,
                                background: data.change > 0 
                                  ? 'linear-gradient(45deg, #4caf50 20%, #66bb6a 80%)'
                                  : data.change < 0
                                    ? 'linear-gradient(45deg, #f44336 20%, #e57373 80%)'
                                    : 'linear-gradient(45deg, #9e9e9e 20%, #757575 80%)'
                              }}>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                  {data.change > 0 ? (
                                    <TrendingUp sx={{ color: 'white', fontSize: 20 }} />
                                  ) : data.change < 0 ? (
                                    <TrendingDown sx={{ color: 'white', fontSize: 20 }} />
                                  ) : (
                                    <CompareArrows sx={{ color: 'white', fontSize: 20 }} />
                                  )}
                                  <Typography variant="body2" sx={{ 
                                    color: 'white',
                                    fontWeight: 600,
                                    textTransform: 'uppercase'
                                  }}>
                                    {data.changeType}
                                  </Typography>
                                </Box>
                                <Typography variant="h6" sx={{ 
                                  fontWeight: 800,
                                  color: 'white'
                                }}>
                                  {data.change > 0 ? '+' : ''}{data.change.toFixed(1)}%
                                </Typography>
                              </Box>
                            </CardContent>
                          </Card>
                        </Grid>
                      );
                    })}
                    </Grid>
                  </CardContent>
                </Card>
              </Zoom>
            ) : (
              <Paper sx={{ 
                p: 6, 
                textAlign: 'center',
                background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
                border: '1px solid #333333',
                borderRadius: 4
              }}>
                <CompareArrows sx={{ 
                  fontSize: 80, 
                  color: '#0071ce', 
                  mb: 3,
                  opacity: 0.6
                }} />
                <Typography variant="h5" sx={{ 
                  fontWeight: 700, 
                  mb: 2
                }}>
                  No Simulation Results
                </Typography>
                <Typography variant="body1" sx={{ 
                  opacity: 0.7,
                  maxWidth: 400,
                  mx: 'auto',
                  lineHeight: 1.6
                }}>
                  Configure scenario parameters and run simulation to see the impact on your forecasts
                </Typography>
              </Paper>
            )}
          </Grid>
        </Grid>
      </Box>
    </Fade>
  );
};

export default SimulationPanel; 