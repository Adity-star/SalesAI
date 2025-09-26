import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Card,
  CardContent,
  Grid,
  Chip,
  LinearProgress
} from '@mui/material';
import {
  Store,
  TrendingUp,
  Assessment
} from '@mui/icons-material';

const StorePerformance = ({ performance, selectedStores }) => {
  if (!performance || !performance.performance || Object.keys(performance.performance).length === 0) {
    return (
      <Paper sx={{ 
        p: 6, 
        textAlign: 'center',
        background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
        border: '1px solid #333333',
        borderRadius: 4
      }}>
        <Store sx={{ 
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
          No Performance Data Available
        </Typography>
        <Typography variant="body1" sx={{ 
          opacity: 0.7,
          maxWidth: 400,
          mx: 'auto',
          lineHeight: 1.6
        }}>
          Select stores and generate forecasts to view performance analytics
        </Typography>
      </Paper>
    );
  }

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        <Assessment sx={{ mr: 1, verticalAlign: 'middle' }} />
        Store Performance Analytics
      </Typography>

      <Grid container spacing={3}>
        {Object.entries(performance.performance).map(([storeId, data]) => (
          <Grid item xs={12} md={6} key={storeId}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  {storeId}
                </Typography>
                
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Total Sales
                    </Typography>
                    <Typography variant="h6" color="primary">
                      {data.total_sales.toLocaleString()}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Avg Daily Sales
                    </Typography>
                    <Typography variant="h6" color="secondary">
                      {Math.round(data.avg_daily_sales).toLocaleString()}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      SKU Count
                    </Typography>
                    <Typography variant="body1">
                      {data.sku_count}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Top SKU
                    </Typography>
                    <Typography variant="body1">
                      {data.top_performing_sku}
                    </Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default StorePerformance; 