import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Grid,
  Paper,
  Typography,
  AppBar,
  Toolbar,
  Card,
  CardContent,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Alert,
  CircularProgress,
  Tabs,
  Tab,
  Divider,
  IconButton,
  Tooltip,
  Fade,
  Zoom
} from '@mui/material';
import {
  TrendingUp,
  Inventory,
  Store,
  Assessment,
  Timeline,
  Warning,
  CheckCircle,
  Error,
  Refresh,
  Settings,
  Analytics,
  Dashboard as DashboardIcon,
  WbSunny,
  Event
} from '@mui/icons-material';
import ForecastChart from './ForecastChart';
import InventorySuggestions from './InventorySuggestions';
import StorePerformance from './StorePerformance';
import SimulationPanel from './SimulationPanel';
import WeatherWidget from './WeatherWidget';
import { fetchForecast, fetchInventorySuggestions, fetchStorePerformance } from '../services/api';
import SimpleChatbot from './SimpleChatbot';

function Dashboard() {
  const [selectedSKUs, setSelectedSKUs] = useState(['SKU001', 'SKU002']);
  const [selectedStores, setSelectedStores] = useState(['STORE001', 'STORE002']);
  const [forecastDays, setForecastDays] = useState(30);
  const [modelType, setModelType] = useState('lstm');
  const [tabValue, setTabValue] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const [forecastData, setForecastData] = useState(null);
  const [inventorySuggestions, setInventorySuggestions] = useState(null);
  const [storePerformance, setStorePerformance] = useState(null);

  const skuOptions = [
    { id: 'SKU001', name: 'Premium Coffee Beans', category: 'Beverages' },
    { id: 'SKU002', name: 'Organic Milk', category: 'Dairy' },
    { id: 'SKU003', name: 'Whole Grain Bread', category: 'Bakery' },
    { id: 'SKU004', name: 'Fresh Bananas', category: 'Produce' },
    { id: 'SKU005', name: 'Chicken Breast', category: 'Meat' }
  ];

  const storeOptions = [
    { id: 'STORE001', name: 'New York', region: 'Northeast' },
    { id: 'STORE002', name: 'Los Angeles', region: 'West Coast' },
    { id: 'STORE003', name: 'Chicago', region: 'Midwest' }
  ];

  const generateForecast = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const forecast = await fetchForecast({
        sku_ids: selectedSKUs,
        store_ids: selectedStores,
        forecast_days: forecastDays,
        model_type: modelType
      });
      
      setForecastData(forecast);
      
      // Also fetch inventory suggestions
      const suggestions = await fetchInventorySuggestions({
        forecasts: forecast.forecasts,
        current_inventory: {
          'SKU001_STORE001': 150,
          'SKU001_STORE002': 200,
          'SKU002_STORE001': 300,
          'SKU002_STORE002': 250
        }
      });
      
      setInventorySuggestions(suggestions);
      
    } catch (err) {
      setError('Failed to generate forecast. Please try again.');
      console.error('Forecast error:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadStorePerformance = async () => {
    try {
      const performance = await fetchStorePerformance(selectedStores, 30);
      setStorePerformance(performance);
    } catch (err) {
      console.error('Performance error:', err);
    }
  };

  useEffect(() => {
    loadStorePerformance();
  }, [selectedStores]);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const getUrgencyColor = (urgency) => {
    switch (urgency) {
      case 'HIGH': return 'error';
      case 'MEDIUM': return 'warning';
      case 'LOW': return 'success';
      default: return 'default';
    }
  };

  return (
    <Box sx={{ flexGrow: 1, minHeight: '100vh', background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)' }}>
      <AppBar position="static" elevation={0}>
        <Toolbar sx={{ minHeight: '80px', px: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mr: 4 }}>
            <Box sx={{ 
              p: 1.5, 
              borderRadius: 3, 
              background: 'linear-gradient(45deg, #ffc220 30%, #ff8f00 90%)',
              mr: 2,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <Store sx={{ fontSize: 28, color: '#000000' }} />
            </Box>
            <Box>
              <Typography variant="h4" component="div" sx={{ 
                fontWeight: 800, 
                background: 'linear-gradient(45deg, #ffffff 30%, #ffc220 90%)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                letterSpacing: '-0.02em',
                mb: -0.5
              }}>
                Walmart AI Forecasting
              </Typography>
              <Typography variant="subtitle2" sx={{ 
                opacity: 0.8, 
                fontWeight: 500,
                letterSpacing: '0.05em'
              }}>
                AI-Driven Demand Forecasting System
              </Typography>
            </Box>
          </Box>
          
          <Box sx={{ flexGrow: 1 }} />
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Tooltip title="System Status">
              <Chip 
                label="Live" 
                color="success" 
                size="small" 
                icon={<CheckCircle />}
                sx={{ 
                  background: 'linear-gradient(45deg, #4caf50 30%, #66bb6a 90%)',
                  color: 'white',
                  fontWeight: 700,
                  fontSize: '0.75rem',
                  height: 28
                }}
              />
            </Tooltip>
            <Chip 
              label="v2.0" 
              variant="outlined" 
              size="small"
              sx={{ 
                borderColor: '#ffc220', 
                color: '#ffc220',
                fontWeight: 600,
                height: 28
              }}
            />
            <Tooltip title="Refresh Data">
              <IconButton 
                onClick={loadStorePerformance}
                sx={{ 
                  color: 'white',
                  '&:hover': { 
                    background: 'rgba(255, 255, 255, 0.1)',
                    transform: 'rotate(180deg)',
                    transition: 'all 0.3s ease'
                  }
                }}
              >
                <Refresh />
              </IconButton>
            </Tooltip>
          </Box>
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" sx={{ mt: 4, mb: 4, px: 3 }}>
        {/* Control Panel */}
        <Fade in timeout={800}>
          <Paper sx={{ 
            p: 4, 
            mb: 4, 
            background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
            border: '1px solid #333333',
            borderRadius: 4,
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
            backdropFilter: 'blur(10px)'
          }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 4 }}>
              <Box sx={{ 
                p: 2, 
                borderRadius: 3, 
                background: 'linear-gradient(45deg, #0071ce 30%, #0056a3 90%)',
                mr: 3,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                <Timeline sx={{ color: 'white', fontSize: 28 }} />
              </Box>
              <Box>
                <Typography variant="h4" sx={{ 
                  fontWeight: 700,
                  mb: 0.5,
                  letterSpacing: '-0.01em'
                }}>
                  Forecast Configuration
                </Typography>
                <Typography variant="body2" sx={{ 
                  opacity: 0.7,
                  fontWeight: 500
                }}>
                  Configure your forecasting parameters and model settings
                </Typography>
              </Box>
            </Box>
            
            <Grid container spacing={3} alignItems="center">
              <Grid item xs={12} md={3}>
                <FormControl fullWidth>
                  <InputLabel sx={{ fontWeight: 600 }}>Select SKUs</InputLabel>
                  <Select
                    multiple
                    value={selectedSKUs}
                    onChange={(e) => setSelectedSKUs(e.target.value)}
                    renderValue={(selected) => (
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                        {selected.map((value) => {
                          const sku = skuOptions.find(s => s.id === value);
                          return (
                            <Chip 
                              key={value} 
                              label={sku?.name || value} 
                              size="small"
                              sx={{ 
                                fontWeight: 600,
                                background: 'linear-gradient(45deg, #0071ce 30%, #0056a3 90%)',
                                color: 'white'
                              }}
                            />
                          );
                        })}
                      </Box>
                    )}
                  >
                    {skuOptions.map((sku) => (
                      <MenuItem key={sku.id} value={sku.id}>
                        <Box>
                          <Typography variant="body2" sx={{ fontWeight: 600 }}>
                            {sku.name}
                          </Typography>
                          <Typography variant="caption" sx={{ opacity: 0.7 }}>
                            {sku.category}
                          </Typography>
                        </Box>
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} md={3}>
                <FormControl fullWidth>
                  <InputLabel sx={{ fontWeight: 600 }}>Select Stores</InputLabel>
                  <Select
                    multiple
                    value={selectedStores}
                    onChange={(e) => setSelectedStores(e.target.value)}
                    renderValue={(selected) => (
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                        {selected.map((value) => {
                          const store = storeOptions.find(s => s.id === value);
                          return (
                            <Chip 
                              key={value} 
                              label={store?.name || value} 
                              size="small"
                              sx={{ 
                                fontWeight: 600,
                                background: 'linear-gradient(45deg, #ffc220 30%, #ff8f00 90%)',
                                color: '#000000'
                              }}
                            />
                          );
                        })}
                      </Box>
                    )}
                  >
                    {storeOptions.map((store) => (
                      <MenuItem key={store.id} value={store.id}>
                        <Box>
                          <Typography variant="body2" sx={{ fontWeight: 600 }}>
                            {store.name}
                          </Typography>
                          <Typography variant="caption" sx={{ opacity: 0.7 }}>
                            {store.region}
                          </Typography>
                        </Box>
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} md={2}>
                <FormControl fullWidth>
                  <InputLabel sx={{ fontWeight: 600 }}>Forecast Period</InputLabel>
                  <Select
                    value={forecastDays}
                    onChange={(e) => setForecastDays(e.target.value)}
                  >
                    <MenuItem value={7}>7 days</MenuItem>
                    <MenuItem value={14}>14 days</MenuItem>
                    <MenuItem value={30}>30 days</MenuItem>
                    <MenuItem value={60}>60 days</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} md={2}>
                <FormControl fullWidth>
                  <InputLabel sx={{ fontWeight: 600 }}>Model Type</InputLabel>
                  <Select
                    value={modelType}
                    onChange={(e) => setModelType(e.target.value)}
                  >
                    <MenuItem value="lstm">LSTM Neural Network</MenuItem>
                    <MenuItem value="arima">ARIMA</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} md={2}>
                <Button
                  variant="contained"
                  onClick={generateForecast}
                  disabled={loading}
                  fullWidth
                  sx={{ 
                    height: 56,
                    fontWeight: 700,
                    fontSize: '0.875rem',
                    background: 'linear-gradient(45deg, #0071ce 30%, #0056a3 90%)',
                    '&:hover': {
                      background: 'linear-gradient(45deg, #0056a3 30%, #004080 90%)',
                      transform: 'translateY(-2px)',
                      boxShadow: '0 8px 25px rgba(0, 113, 206, 0.4)'
                    },
                    '&:disabled': {
                      background: '#666666'
                    }
                  }}
                >
                  {loading ? (
                    <CircularProgress size={24} sx={{ color: 'white' }} />
                  ) : (
                    <>
                      <TrendingUp sx={{ mr: 1, fontSize: 20 }} />
                      Generate Forecast
                    </>
                  )}
                </Button>
              </Grid>
            </Grid>
          </Paper>
        </Fade>

        {/* Error Alert */}
        {error && (
          <Fade in timeout={500}>
            <Alert 
              severity="error" 
              sx={{ 
                mb: 3, 
                borderRadius: 2,
                fontWeight: 600,
                '& .MuiAlert-icon': { fontSize: 28 }
              }}
              onClose={() => setError(null)}
            >
              {error}
            </Alert>
          </Fade>
        )}

        {/* Main Content Tabs */}
        <Paper sx={{ 
          background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
          border: '1px solid #333333',
          borderRadius: 4,
          overflow: 'hidden',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
          backdropFilter: 'blur(10px)'
        }}>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs 
              value={tabValue} 
              onChange={handleTabChange} 
              sx={{ 
                px: 3,
                '& .MuiTab-root': {
                  minHeight: 64,
                  fontSize: '0.875rem',
                  fontWeight: 600,
                  textTransform: 'none'
                }
              }}
            >
              <Tab 
                icon={<DashboardIcon />} 
                label="Dashboard Overview" 
                iconPosition="start"
              />
              <Tab 
                icon={<TrendingUp />} 
                label="Forecast Analytics" 
                iconPosition="start"
              />
              <Tab 
                icon={<WbSunny />} 
                label="Weather & Events" 
                iconPosition="start"
              />
              <Tab 
                icon={<Inventory />} 
                label="Inventory Management" 
                iconPosition="start"
              />
              <Tab 
                icon={<Store />} 
                label="Store Performance" 
                iconPosition="start"
              />
              <Tab 
                icon={<Assessment />} 
                label="Simulation Tools" 
                iconPosition="start"
              />
            </Tabs>
          </Box>

          <TabPanel value={tabValue} index={0}>
            <Grid container spacing={3}>
              <Grid item xs={12} lg={8}>
                <Card sx={{ height: '100%' }}>
                  <CardContent>
                    <Typography variant="h5" sx={{ mb: 3, fontWeight: 700 }}>
                      Forecast Overview
                    </Typography>
                    {forecastData ? (
                      <ForecastChart data={forecastData} />
                    ) : (
                      <Box sx={{ 
                        display: 'flex', 
                        flexDirection: 'column', 
                        alignItems: 'center', 
                        justifyContent: 'center',
                        height: 400,
                        opacity: 0.6
                      }}>
                        <Analytics sx={{ fontSize: 64, mb: 2, color: '#0071ce' }} />
                        <Typography variant="h6" sx={{ mb: 1, fontWeight: 600 }}>
                          No Forecast Data
                        </Typography>
                        <Typography variant="body2" sx={{ textAlign: 'center' }}>
                          Generate a forecast to view analytics and insights
                        </Typography>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} lg={4}>
                <Grid container spacing={3}>
                  <Grid item xs={12}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" sx={{ mb: 2, fontWeight: 700 }}>
                          Quick Stats
                        </Typography>
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                          <Box sx={{ 
                            display: 'flex', 
                            justifyContent: 'space-between', 
                            alignItems: 'center',
                            p: 2,
                            borderRadius: 2,
                            background: 'linear-gradient(45deg, #0071ce 20%, #0056a3 80%)',
                            color: 'white'
                          }}>
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                              Selected SKUs
                            </Typography>
                            <Typography variant="h6" sx={{ fontWeight: 700 }}>
                              {selectedSKUs.length}
                            </Typography>
                          </Box>
                          <Box sx={{ 
                            display: 'flex', 
                            justifyContent: 'space-between', 
                            alignItems: 'center',
                            p: 2,
                            borderRadius: 2,
                            background: 'linear-gradient(45deg, #ffc220 20%, #ff8f00 80%)',
                            color: '#000000'
                          }}>
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                              Selected Stores
                            </Typography>
                            <Typography variant="h6" sx={{ fontWeight: 700 }}>
                              {selectedStores.length}
                            </Typography>
                          </Box>
                          <Box sx={{ 
                            display: 'flex', 
                            justifyContent: 'space-between', 
                            alignItems: 'center',
                            p: 2,
                            borderRadius: 2,
                            background: 'linear-gradient(45deg, #4caf50 20%, #388e3c 80%)',
                            color: 'white'
                          }}>
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                              Forecast Days
                            </Typography>
                            <Typography variant="h6" sx={{ fontWeight: 700 }}>
                              {forecastDays}
                            </Typography>
                          </Box>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>

                  {/* Simplified Weather Widget */}
                  <Grid item xs={12}>
                    <Card>
                      <CardContent>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                          <WbSunny sx={{ mr: 1, color: '#0071ce' }} />
                          <Typography variant="h6" sx={{ fontWeight: 700 }}>
                            Current Weather
                          </Typography>
                        </Box>
                        <Grid container spacing={2}>
                          {selectedStores.slice(0, 2).map((storeId) => (
                            <Grid item xs={6} key={storeId}>
                              <Box sx={{ 
                                p: 2, 
                                borderRadius: 2, 
                                background: 'rgba(255, 255, 255, 0.05)',
                                border: '1px solid #333333'
                              }}>
                                <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
                                  {storeId}
                                </Typography>
                                <Typography variant="h5" sx={{ fontWeight: 700, color: '#0071ce' }}>
                                  {Math.floor(Math.random() * 20) + 15}°C
                                </Typography>
                                <Typography variant="caption" sx={{ opacity: 0.7 }}>
                                  Clear
                                </Typography>
                              </Box>
                            </Grid>
                          ))}
                        </Grid>
                      </CardContent>
                    </Card>
                  </Grid>

                  {/* Simplified Events Widget */}
                  <Grid item xs={12}>
                    <Card>
                      <CardContent>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                          <Event sx={{ mr: 1, color: '#e74c3c' }} />
                          <Typography variant="h6" sx={{ fontWeight: 700 }}>
                            Upcoming Events
                          </Typography>
                        </Box>
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                          <Box sx={{ 
                            display: 'flex', 
                            justifyContent: 'space-between', 
                            alignItems: 'center',
                            p: 1.5,
                            borderRadius: 1,
                            background: 'rgba(231, 76, 60, 0.1)',
                            border: '1px solid rgba(231, 76, 60, 0.3)'
                          }}>
                            <Box>
                              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                Christmas
                              </Typography>
                              <Typography variant="caption" sx={{ opacity: 0.7 }}>
                                Dec 25, 2024
                              </Typography>
                            </Box>
                            <Chip 
                              label="2.5x" 
                              size="small" 
                              sx={{ 
                                background: '#e74c3c', 
                                color: 'white',
                                fontWeight: 600
                              }} 
                            />
                          </Box>
                          <Box sx={{ 
                            display: 'flex', 
                            justifyContent: 'space-between', 
                            alignItems: 'center',
                            p: 1.5,
                            borderRadius: 1,
                            background: 'rgba(52, 152, 219, 0.1)',
                            border: '1px solid rgba(52, 152, 219, 0.3)'
                          }}>
                            <Box>
                              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                New Year's Eve
                              </Typography>
                              <Typography variant="caption" sx={{ opacity: 0.7 }}>
                                Dec 31, 2024
                              </Typography>
                            </Box>
                            <Chip 
                              label="1.7x" 
                              size="small" 
                              sx={{ 
                                background: '#3498db', 
                                color: 'white',
                                fontWeight: 600
                              }} 
                            />
                          </Box>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              </Grid>
            </Grid>
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            {forecastData ? (
              <ForecastChart data={forecastData} />
            ) : (
              <Box sx={{ 
                display: 'flex', 
                flexDirection: 'column', 
                alignItems: 'center', 
                justifyContent: 'center',
                height: 400,
                opacity: 0.6
              }}>
                <TrendingUp sx={{ fontSize: 64, mb: 2, color: '#0071ce' }} />
                <Typography variant="h6" sx={{ mb: 1, fontWeight: 600 }}>
                  No Forecast Data Available
                </Typography>
                <Typography variant="body2" sx={{ textAlign: 'center', mb: 3 }}>
                  Generate a forecast to view detailed analytics and trends
                </Typography>
                <Button
                  variant="contained"
                  onClick={generateForecast}
                  sx={{ 
                    fontWeight: 600,
                    background: 'linear-gradient(45deg, #0071ce 30%, #0056a3 90%)'
                  }}
                >
                  Generate Forecast
                </Button>
              </Box>
            )}
          </TabPanel>

          <TabPanel value={tabValue} index={2}>
            <WeatherWidget selectedStores={selectedStores} selectedSKUs={selectedSKUs} />
          </TabPanel>

          <TabPanel value={tabValue} index={3}>
            <InventorySuggestions suggestions={inventorySuggestions} loading={loading} />
          </TabPanel>

          <TabPanel value={tabValue} index={4}>
            <StorePerformance performance={storePerformance} selectedStores={selectedStores} />
          </TabPanel>

          <TabPanel value={tabValue} index={5}>
            <SimulationPanel forecastData={forecastData} />
          </TabPanel>
        </Paper>
      </Container>
      <div className="chatbot-container">
        <SimpleChatbot />
      </div>
    </Box>
  );
}

function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

export default Dashboard;
