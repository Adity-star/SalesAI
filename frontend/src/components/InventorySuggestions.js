import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Card,
  CardContent,
  Grid,
  Chip,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Button,
  LinearProgress
} from '@mui/material';
import {
  Inventory,
  Warning,
  CheckCircle,
  Error,
  TrendingUp,
  Schedule
} from '@mui/icons-material';

const InventorySuggestions = ({ suggestions, loading }) => {
  if (loading) {
    return (
      <Box>
        <LinearProgress />
        <Typography variant="body2" sx={{ mt: 1 }}>
          Generating inventory suggestions...
        </Typography>
      </Box>
    );
  }

  if (!suggestions || !suggestions.suggestions || Object.keys(suggestions.suggestions).length === 0) {
    return (
      <Paper sx={{ 
        p: 6, 
        textAlign: 'center',
        background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
        border: '1px solid #333333',
        borderRadius: 4
      }}>
        <Inventory sx={{ 
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
          No Inventory Suggestions Available
        </Typography>
        <Typography variant="body1" sx={{ 
          opacity: 0.7,
          maxWidth: 400,
          mx: 'auto',
          lineHeight: 1.6
        }}>
          Generate a forecast to see inventory recommendations and optimization suggestions
        </Typography>
      </Paper>
    );
  }

  const getUrgencyColor = (urgency) => {
    switch (urgency) {
      case 'HIGH': return 'error';
      case 'MEDIUM': return 'warning';
      case 'LOW': return 'success';
      default: return 'default';
    }
  };

  const getUrgencyIcon = (urgency) => {
    switch (urgency) {
      case 'HIGH': return <Error />;
      case 'MEDIUM': return <Warning />;
      case 'LOW': return <CheckCircle />;
      default: return <Schedule />;
    }
  };

  const calculateStockoutRisk = (daysUntilStockout) => {
    if (daysUntilStockout === 0) return 'IMMINENT';
    if (daysUntilStockout <= 3) return 'HIGH';
    if (daysUntilStockout <= 7) return 'MEDIUM';
    return 'LOW';
  };

  const getStockoutRiskColor = (risk) => {
    switch (risk) {
      case 'IMMINENT': return 'error';
      case 'HIGH': return 'error';
      case 'MEDIUM': return 'warning';
      case 'LOW': return 'success';
      default: return 'default';
    }
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        <Inventory sx={{ mr: 1, verticalAlign: 'middle' }} />
        Inventory Management Recommendations
      </Typography>

      <Grid container spacing={3}>
        {/* Summary Cards */}
        <Grid item xs={12}>
          <Grid container spacing={2}>
            {Object.entries(suggestions.suggestions).map(([key, suggestion]) => (
              <Grid item xs={12} md={6} lg={4} key={key}>
                <Card>
                  <CardContent>
                    <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                      <Typography variant="h6">
                        {suggestion.sku_id} - {suggestion.store_id}
                      </Typography>
                      <Chip
                        icon={getUrgencyIcon(suggestion.urgency)}
                        label={suggestion.urgency}
                        color={getUrgencyColor(suggestion.urgency)}
                        size="small"
                      />
                    </Box>

                    <Grid container spacing={2}>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">
                          Current Stock
                        </Typography>
                        <Typography variant="h6" color="primary">
                          {suggestion.current_inventory}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">
                          Forecasted Demand
                        </Typography>
                        <Typography variant="h6" color="secondary">
                          {suggestion.forecasted_demand}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">
                          Suggested Reorder
                        </Typography>
                        <Typography variant="h6" color="success.main">
                          {suggestion.suggested_reorder}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">
                          Safety Stock
                        </Typography>
                        <Typography variant="body1">
                          {suggestion.safety_stock}
                        </Typography>
                      </Grid>
                    </Grid>

                    <Box mt={2}>
                      <Typography variant="body2" color="text.secondary">
                        Stockout Risk
                      </Typography>
                      <Chip
                        label={`${suggestion.days_until_stockout} days`}
                        color={getStockoutRiskColor(calculateStockoutRisk(suggestion.days_until_stockout))}
                        size="small"
                        sx={{ mt: 1 }}
                      />
                    </Box>

                    <Box mt={2}>
                      <Typography variant="body2" color="text.secondary">
                        Recommended Action
                      </Typography>
                      <Typography variant="body1" fontWeight="bold" color={getUrgencyColor(suggestion.urgency)}>
                        {suggestion.action}
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Grid>

        {/* Detailed Table */}
        <Grid item xs={12}>
          <Paper>
            <Typography variant="h6" sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
              Detailed Inventory Analysis
            </Typography>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>SKU - Store</TableCell>
                    <TableCell align="right">Current Stock</TableCell>
                    <TableCell align="right">Forecasted Demand</TableCell>
                    <TableCell align="right">Suggested Reorder</TableCell>
                    <TableCell align="center">Urgency</TableCell>
                    <TableCell align="center">Stockout Risk</TableCell>
                    <TableCell align="center">Action</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Object.entries(suggestions.suggestions).map(([key, suggestion]) => (
                    <TableRow key={key}>
                      <TableCell>
                        <Typography variant="body2" fontWeight="bold">
                          {suggestion.sku_id} - {suggestion.store_id}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Typography variant="body2">
                          {suggestion.current_inventory}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Typography variant="body2">
                          {suggestion.forecasted_demand}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Typography variant="body2" color="success.main" fontWeight="bold">
                          {suggestion.suggested_reorder}
                        </Typography>
                      </TableCell>
                      <TableCell align="center">
                        <Chip
                          label={suggestion.urgency}
                          color={getUrgencyColor(suggestion.urgency)}
                          size="small"
                        />
                      </TableCell>
                      <TableCell align="center">
                        <Chip
                          label={`${suggestion.days_until_stockout} days`}
                          color={getStockoutRiskColor(calculateStockoutRisk(suggestion.days_until_stockout))}
                          size="small"
                        />
                      </TableCell>
                      <TableCell align="center">
                        <Button
                          variant="outlined"
                          size="small"
                          color={getUrgencyColor(suggestion.urgency)}
                        >
                          {suggestion.action}
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>

        {/* Key Insights */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                <TrendingUp sx={{ mr: 1, verticalAlign: 'middle' }} />
                Key Insights
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12} md={4}>
                  <Alert severity="info">
                    <Typography variant="body2">
                      <strong>High Priority Items:</strong> {Object.values(suggestions.suggestions).filter(s => s.urgency === 'HIGH').length} items need immediate attention
                    </Typography>
                  </Alert>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Alert severity="warning">
                    <Typography variant="body2">
                      <strong>Stockout Risk:</strong> {Object.values(suggestions.suggestions).filter(s => s.days_until_stockout <= 7).length} items at risk within 7 days
                    </Typography>
                  </Alert>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Alert severity="success">
                    <Typography variant="body2">
                      <strong>Total Reorder Value:</strong> ${Object.values(suggestions.suggestions).reduce((sum, s) => sum + s.suggested_reorder, 0).toLocaleString()}
                    </Typography>
                  </Alert>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default InventorySuggestions; 