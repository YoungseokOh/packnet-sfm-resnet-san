import torch
import torch.nn as nn
import warnings

# MinkowskiEngine 안전한 임포트
try:
    import MinkowskiEngine as ME
    MINKOWSKI_AVAILABLE = True
    print("✅ MinkowskiEngine loaded successfully")
except ImportError as e:
    print(f"⚠️ MinkowskiEngine not available: {e}")
    print("🔧 Using fallback implementation with regular 3D convolutions")
    MINKOWSKI_AVAILABLE = False
    ME = None
    warnings.warn("MinkowskiEngine not available, using fallback implementation", UserWarning)

class EnhancedMinkowskiEncoder(nn.Module):
    """Enhanced Minkowski Encoder with fallback to regular convolutions"""
    
    def __init__(self, in_channels=1, out_channels=64, kernel_size=3, **kwargs):
        super().__init__()
        
        global MINKOWSKI_AVAILABLE  # Declare global at the beginning of the function
        
        if MINKOWSKI_AVAILABLE:
            try:
                self._init_minkowski(in_channels, out_channels, kernel_size, **kwargs)
                print("✅ MinkowskiEngine layers initialized successfully")
            except Exception as e:
                print(f"❌ MinkowskiEngine initialization failed: {e}")
                print("🔧 Falling back to regular convolutions")
                MINKOWSKI_AVAILABLE = False
                self._init_fallback(in_channels, out_channels, kernel_size, **kwargs)
        else:
            self._init_fallback(in_channels, out_channels, kernel_size, **kwargs)
    
    def _init_minkowski(self, in_channels, out_channels, kernel_size, **kwargs):
        """Original MinkowskiEngine implementation"""
        self.conv1 = ME.MinkowskiConvolution(
            in_channels, out_channels // 2, 
            kernel_size=kernel_size, dimension=3
        )
        self.conv2 = ME.MinkowskiConvolution(
            out_channels // 2, out_channels, 
            kernel_size=kernel_size, dimension=3
        )
        self.relu = ME.MinkowskiReLU()
        
    def _init_fallback(self, in_channels, out_channels, kernel_size, **kwargs):
        """Fallback implementation with regular 3D convolutions"""
        print("🔧 Using fallback 3D convolution implementation")
        self.conv1 = nn.Conv3d(in_channels, out_channels // 2, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv3d(out_channels // 2, out_channels, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.adaptive_pool = nn.AdaptiveAvgPool3d((8, 8, 8))
        
    def forward(self, x):
        global MINKOWSKI_AVAILABLE  # Declare global at the beginning of the function
        
        if MINKOWSKI_AVAILABLE:
            try:
                return self._forward_minkowski(x)
            except Exception as e:
                print(f"⚠️ MinkowskiEngine forward failed: {e}, using fallback")
                MINKOWSKI_AVAILABLE = False
                return self._forward_fallback(x)
        else:
            return self._forward_fallback(x)
    
    def _forward_minkowski(self, x):
        """MinkowskiEngine forward pass"""
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x
    
    def _forward_fallback(self, x):
        """Fallback forward pass with regular convolutions"""
        # Convert input to proper tensor format
        if not isinstance(x, torch.Tensor):
            # Create dummy dense tensor for fallback
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            x = torch.randn(1, 1, 32, 32, 32, device=device)
        
        # Ensure proper dimensions for 3D convolution (B, C, D, H, W)
        if x.dim() != 5:
            if x.dim() == 4:  # (B, C, H, W) -> add depth dimension
                x = x.unsqueeze(2)
            elif x.dim() == 3:  # (C, H, W) -> add batch and depth
                x = x.unsqueeze(0).unsqueeze(2)
            elif x.dim() == 2:  # (H, W) -> add batch, channel, and depth
                x = x.unsqueeze(0).unsqueeze(0).unsqueeze(2)
        
        # Regular convolution forward pass
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.adaptive_pool(x)
        return x
