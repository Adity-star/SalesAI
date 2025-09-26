import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'https://walmart-forecast.onrender.com/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const fetchForecast = async (params) => {
  try {
    console.log('Making forecast request to:', `${API_BASE_URL}/forecast`);
    console.log('Request params:', params);
    
    const response = await api.post('/forecast', params);
    console.log('Forecast response:', response.data);
    return response.data;
  } catch (error) {
    console.error('Forecast API error details:', {
      message: error.message,
      status: error.response?.status,
      statusText: error.response?.statusText,
      data: error.response?.data,
      config: error.config
    });
    throw error;
  }
};

export const fetchInventorySuggestions = async (params) => {
  try {
    const response = await api.post('/inventory-suggestions', params);
    return response.data;
  } catch (error) {
    console.error('Inventory suggestions API error:', error);
    throw error;
  }
};

export const fetchStorePerformance = async (storeIds, days = 30) => {
  try {
    const response = await api.get('/store-performance', {
      params: {
        store_ids: storeIds.join(','),
        days: days
      }
    });
    return response.data;
  } catch (error) {
    console.error('Store performance API error:', error);
    throw error;
  }
};

export const fetchSKUAnalytics = async (skuId, storeId, days = 90) => {
  try {
    const response = await api.get('/sku-analytics', {
      params: {
        sku_id: skuId,
        store_id: storeId,
        days: days
      }
    });
    return response.data;
  } catch (error) {
    console.error('SKU analytics API error:', error);
    throw error;
  }
};

export const fetchWeatherData = async (storeIds) => {
  try {
    const response = await api.get('/weather-data', {
      params: {
        store_ids: storeIds.join(',')
      }
    });
    return response.data;
  } catch (error) {
    console.error('Weather API error:', error);
    throw error;
  }
};

export const runSimulation = async (params) => {
  try {
    const response = await api.post('/simulation', params);
    return response.data;
  } catch (error) {
    console.error('Simulation API error:', error);
    throw error;
  }
};

export const healthCheck = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    console.error('Health check error:', error);
    throw error;
  }
}; 