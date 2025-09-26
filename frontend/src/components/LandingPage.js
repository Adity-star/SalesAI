
import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Container,
  Box,
  Grid,
  Paper,
  Icon
} from '@mui/material';
import { 
    TrendingUp, 
    Inventory, 
    Store, 
    ArrowForward, 
    Analytics, 
    ModelTraining, 
    WbSunny 
} from '@mui/icons-material';

const FeatureCard = ({ icon, title, description }) => (
    <Paper elevation={3} sx={{
        p: 4,
        height: '100%',
        background: 'linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%)',
        border: '1px solid #333',
        borderRadius: 3,
        textAlign: 'center',
        color: 'white'
    }}>
        <Box sx={{ color: '#0071ce', mb: 2 }}>
            {icon}
        </Box>
        <Typography variant="h5" component="h3" sx={{ fontWeight: 700, mb: 2 }}>
            {title}
        </Typography>
        <Typography variant="body1" sx={{ opacity: 0.7 }}>
            {description}
        </Typography>
    </Paper>
);

const LandingPage = () => {
  return (
    <Box sx={{ minHeight: '100vh', background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)', color: 'white' }}>
      {/* Header */}
      <AppBar position="static" color="transparent" elevation={0} sx={{ borderBottom: '1px solid #333' }}>
        <Toolbar sx={{ justifyContent: 'space-between', py: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Store sx={{ fontSize: 32, color: '#ffc220', mr: 1.5 }} />
                <Typography variant="h5" component="div" sx={{ fontWeight: 800 }}>
                    Peace AI
                </Typography>
            </Box>
          <Button color="inherit" component={RouterLink} to="/login" sx={{ fontWeight: 600 }}>
            Login
          </Button>
        </Toolbar>
      </AppBar>

      {/* Hero Section */}
      <Container maxWidth="md" sx={{ textAlign: 'center', py: { xs: 8, md: 16 } }}>
        <Typography variant="h2" component="h1" sx={{
            fontWeight: 800,
            mb: 3,
            background: 'linear-gradient(45deg, #ffffff 30%, #ffc220 90%)',
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent'
        }}>
          Revolutionize Your Retail with AI-Powered Forecasting
        </Typography>
        <Typography variant="h6" component="p" sx={{ mb: 5, opacity: 0.7, maxWidth: '700px', mx: 'auto' }}>
          Leverage cutting-edge AI to predict demand, optimize inventory, and boost performance across all your stores.
        </Typography>
        <Button 
            variant="contained" 
            size="large" 
            component={RouterLink} 
            to="/dashboard"
            endIcon={<ArrowForward />}
            sx={{
                py: 1.5,
                px: 4,
                fontWeight: 700,
                fontSize: '1rem',
                borderRadius: 2,
                background: 'linear-gradient(45deg, #0071ce 30%, #0056a3 90%)',
                '&:hover': {
                    background: 'linear-gradient(45deg, #0056a3 30%, #004080 90%)',
                    transform: 'translateY(-2px)'
                }
            }}
        >
          Access Your Dashboard
        </Button>
      </Container>

      {/* Features Section */}
      <Box sx={{ background: '#0a0a0a', py: { xs: 8, md: 12 } }}>
        <Container maxWidth="lg">
          <Typography variant="h4" component="h2" sx={{ textAlign: 'center', fontWeight: 700, mb: 8 }}>
            An All-in-One Platform for Intelligent Retail
          </Typography>
          <Grid container spacing={4}>
            <Grid item xs={12} md={4}>
              <FeatureCard 
                icon={<TrendingUp sx={{ fontSize: 48 }}/>} 
                title="Demand Forecasting"
                description="Utilize advanced LSTM and ARIMA models to generate highly accurate, SKU-level demand forecasts for any period."
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <FeatureCard 
                icon={<Inventory sx={{ fontSize: 48 }}/>} 
                title="Inventory Optimization"
                description="Receive intelligent reorder suggestions and safety stock levels to prevent stockouts and reduce holding costs."
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <FeatureCard 
                icon={<Analytics sx={{ fontSize: 48 }}/>} 
                title="Performance Analytics"
                description="Monitor key store metrics, track sales performance, and identify top-performing products and locations instantly."
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <FeatureCard 
                icon={<ModelTraining sx={{ fontSize: 48 }}/>} 
                title="Simulation Tools"
                description="Simulate different scenarios and their potential impact on your sales and inventory before making real-world decisions."
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <FeatureCard 
                icon={<WbSunny sx={{ fontSize: 48 }}/>} 
                title="External Factors"
                description="Automatically factor in weather, holidays, and other external events to further refine forecast accuracy."
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <FeatureCard 
                icon={<Store sx={{ fontSize: 48 }}/>} 
                title="Multi-Store Management"
                description="Seamlessly manage and compare forecasts and performance across multiple stores and regions from one central dashboard."
              />
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* Footer */}
      <Box component="footer" sx={{ textAlign: 'center', py: 5, borderTop: '1px solid #333' }}>
        <Typography variant="body2" sx={{ opacity: 0.6 }}>
          © {new Date().getFullYear()} Peace AI Forecasting Platform. All Rights Reserved.
        </Typography>
      </Box>
    </Box>
  );
};

export default LandingPage;
