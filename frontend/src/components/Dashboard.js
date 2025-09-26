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
  Logout,
  WbSunny,
  Event
} from '@mui/icons-material';
import ForecastChart from './ForecastChart';
import InventorySuggestions from './InventorySuggestions';
import StorePerformance from './StorePerformance';
import SimulationPanel from './SimulationPanel';
import WeatherWidget from './WeatherWidget';
import { Link as RouterLink } from 'react-router-dom';
import { fetchForecast, fetchInventorySuggestions, fetchStorePerformance } from '../services/api';
import AdvancedChatbot from './SimpleChatbot';
import { Home } from '@mui/icons-material'; // Import Home icon

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

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static" elevation={0}>
        <Toolbar sx={{ minHeight: '80px', px: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mr: 4 }}>
            <Store sx={{ fontSize: 32, color: 'secondary.main', mr: 1.5 }} />
            <Typography variant="h5" component="div" sx={{ fontWeight: 800 }}>
                Walmart AI Forecasting
            </Typography>
          </Box>
          
          <Box sx={{ flexGrow: 1 }} />
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Tooltip title="Home">
              <IconButton component={RouterLink} to="/" sx={{ color: 'text.primary', '&:hover': { color: 'primary.main' } }}>
                <Home />
              </IconButton>
            </Tooltip>
            <Chip label="v2.0" variant="outlined" size="small" color="secondary"/>
            <Tooltip title="Refresh Data">
              <IconButton onClick={loadStorePerformance} sx={{ color: 'text.primary', '&:hover': { color: 'primary.main' } }}>
                <Refresh />
              </IconButton>
            </Tooltip>
          </Box>
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" sx={{ mt: 4, mb: 4, px: 3 }}>
        {/* Control Panel */}
        <Paper sx={{ p: 4, mb: 4 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
              <Timeline sx={{ color: 'primary.main', fontSize: 32, mr: 2 }} />
              <Typography variant="h4">Forecast Configuration</Typography>
            </Box>
            
            <Grid container spacing={3} alignItems="center">
              <Grid item xs={12} md={3}>
                <FormControl fullWidth>
                  <InputLabel>Select SKUs</InputLabel>
                  <Select multiple value={selectedSKUs} onChange={(e) => setSelectedSKUs(e.target.value)}>
                    {skuOptions.map((sku) => (
                      <MenuItem key={sku.id} value={sku.id}>{sku.name}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} md={3}>
                <FormControl fullWidth>
                  <InputLabel>Select Stores</InputLabel>
                  <Select multiple value={selectedStores} onChange={(e) => setSelectedStores(e.target.value)}>
                    {storeOptions.map((store) => (
                      <MenuItem key={store.id} value={store.id}>{store.name}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} md={2}>
                <FormControl fullWidth>
                  <InputLabel>Forecast Period</InputLabel>
                  <Select value={forecastDays} onChange={(e) => setForecastDays(e.target.value)}>
                    <MenuItem value={7}>7 days</MenuItem>
                    <MenuItem value={14}>14 days</MenuItem>
                    <MenuItem value={30}>30 days</MenuItem>
                    <MenuItem value={60}>60 days</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} md={2}>
                <FormControl fullWidth>
                  <InputLabel>Model Type</InputLabel>
                  <Select value={modelType} onChange={(e) => setModelType(e.target.value)}>
                    <MenuItem value="lstm">LSTM Neural Network</MenuItem>
                    <MenuItem value="arima">ARIMA</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} md={2}>
                <Button
                  variant="contained"
                  color="primary"
                  onClick={generateForecast}
                  disabled={loading}
                  fullWidth
                  sx={{ height: 56 }}
                >
                  {loading ? <CircularProgress size={24} /> : 'Generate Forecast'}
                </Button>
              </Grid>
            </Grid>
          </Paper>

        {error && (
          <Fade in timeout={500}>
            <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
              {error}
            </Alert>
          </Fade>
        )}

        <Paper>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs value={tabValue} onChange={handleTabChange} aria-label="dashboard tabs">
              <Tab icon={<DashboardIcon />} label="Dashboard" iconPosition="start"/>
              <Tab icon={<TrendingUp />} label="Analytics" iconPosition="start"/>
              <Tab icon={<WbSunny />} label="External Factors" iconPosition="start"/>
              <Tab icon={<Inventory />} label="Inventory" iconPosition="start"/>
              <Tab icon={<Store />} label="Performance" iconPosition="start"/>
              <Tab icon={<Assessment />} label="Simulation" iconPosition="start"/>
            </Tabs>
          </Box>

          <TabPanel value={tabValue} index={0}>
            <Grid container spacing={3}>
              <Grid item xs={12} lg={8}>
                <Card sx={{ height: '100%' }}>
                  <CardContent>
                    {forecastData ? (
                      <ForecastChart data={forecastData} />
                    ) : (
                      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: 400, opacity: 0.6 }}>
                        <Analytics sx={{ fontSize: 64, mb: 2, color: 'primary.main' }} />
                        <Typography variant="h6">No Forecast Data</Typography>
                        <Typography>Generate a forecast to view analytics</Typography>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} lg={4}>
                 <Card>
                    <CardContent>
                        <Typography variant="h6" sx={{ mb: 2}}>Quick Stats</Typography>
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                            <Paper variant="outlined" sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                                <Typography>Selected SKUs</Typography>
                                <Typography variant="h6">{selectedSKUs.length}</Typography>
                            </Paper>
                            <Paper variant="outlined" sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                                <Typography>Selected Stores</Typography>
                                <Typography variant="h6">{selectedStores.length}</Typography>
                            </Paper>
                            <Paper variant="outlined" sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                                <Typography>Forecast Days</Typography>
                                <Typography variant="h6">{forecastDays}</Typography>
                            </Paper>
                        </Box>
                    </CardContent>
                 </Card>
              </Grid>
            </Grid>
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            <ForecastChart data={forecastData} />
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
        <AdvancedChatbot />
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
