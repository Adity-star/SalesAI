
import { createTheme } from '@mui/material/styles';

// A custom theme for the AI dashboard
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00E5FF', // A vibrant, futuristic cyan
    },
    secondary: {
      main: '#FFC220', // Walmart Yellow for specific accents
    },
    background: {
      default: '#0A1929', // A deep, dark blue
      paper: 'rgba(10, 25, 41, 0.7)', // A semi-transparent, glass-like paper color
    },
    text: {
      primary: '#E0E0E0',
      secondary: '#B0B0B0',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 700,
      letterSpacing: '0.02em',
    },
    h5: {
      fontWeight: 600,
    },
    button: {
      textTransform: 'none',
      fontWeight: 600,
    },
  },
  components: {
    // Global component overrides
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none', // Remove default MUI gradients
          border: '1px solid rgba(255, 255, 255, 0.12)',
          backdropFilter: 'blur(10px)',
        },
      },
    },
    MuiCard: {
        styleOverrides: {
            root: {
                background: 'linear-gradient(135deg, rgba(10, 25, 41, 0.6) 0%, rgba(15, 30, 50, 0.7) 100%)',
                transition: 'transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out',
                '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: '0 10px 20px rgba(0, 229, 255, 0.1)',
                },
            },
        },
    },
    MuiButton: {
      styleOverrides: {
        containedPrimary: {
          background: 'linear-gradient(45deg, #00E5FF 30%, #00B8D4 90%)',
          boxShadow: '0 3px 5px 2px rgba(0, 229, 255, .3)',
          color: '#0A1929',
          '&:hover': {
            boxShadow: '0 5px 10px 4px rgba(0, 229, 255, .4)',
          }
        },
      },
    },
    MuiAppBar: {
        styleOverrides: {
            root: {
                background: 'rgba(10, 25, 41, 0.8)',
                borderBottom: '1px solid rgba(255, 255, 255, 0.12)',
                backdropFilter: 'blur(10px)',
            }
        }
    }
  },
});

export default theme;
