#------------------- Imported Same libraries------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import trunc_normal_
#--------------------------------------------------------------------------------------

# ----------------------Same, but few lines are not included--------------------------
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5): #detach_grad=False
        super(LayerNorm, self).__init__()
        
        # There three lines are inserted.
        self.eps = eps
        
        # This line is not included
        #self.detach_grad = detach_grad

        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

        # Here few (6) lines are not included 
        #self.meta1 = nn.Conv2d(1, dim, 1)
		#self.meta2 = nn.Conv2d(1, dim, 1)
		#trunc_normal_(self.meta1.weight, std=.02)
		#nn.init.constant_(self.meta1.bias, 1)
		#trunc_normal_(self.meta2.weight, std=.02)
		#nn.init.constant_(self.meta2.bias, 0)

    def forward(self, input):
        # same line 
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        # Same line
        std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

        # same line
        normalized_input = (input - mean) / std

        # These lines are not included.
        # Here there was few lines that are skipped here. bcz it does not inseretd detached as paramters
        #if self.detach_grad:
		#	rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
		#else:
		#	rescale, rebias = self.meta1(std), self.meta2(mean)

        # Same line
        out = normalized_input * self.weight + self.bias
        return out


# ----------------------Same, but few lines are not included--------------------------
class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        # Same these three lines
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.network_depth = network_depth

    # ------------------------Same Lines-----------------------------------
        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            
            #this line is extra included
            nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features),
            
            nn.ReLU(True),
            nn.Conv2d(hidden_features, out_features, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1 / 4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)
    #-----------------------------------------------------------------------------------------------


# this classs is different, Re-check the model 
class CEFN(nn.Module):
    def __init__(self, dim, network_depth, hidden_features=None, out_features=None):
        super(CEFN, self).__init__()
        self.mlp = Mlp(network_depth=network_depth, in_features=dim, hidden_features=hidden_features,
                       out_features=out_features)
        self.norm = LayerNorm(dim, eps=1e-5)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 8, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.scaler = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        attn = self.scaler * self.ca(x)
        x = self.norm(self.mlp(x))
        return x * attn


# this classs is different, TransformerBlock is used there
# -------------------------------These lines are different, you may check TransformerBlock Further
class LKDBlock(nn.Module):
    def __init__(self, network_depth, dim, mlp_ratio=4.):
        super().__init__()

        # DLKCB
        self.norm1 = nn.BatchNorm2d(dim)
        self.Linear1 = nn.Conv2d(dim, dim, 1)
        self.DWConv = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, padding_mode='reflect')
        self.DWDConv = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3, padding_mode='reflect')
        self.Linear2 = nn.Conv2d(dim, dim, 1)

        # CEFN
        self.norm2 = nn.BatchNorm2d(dim)
        self.cemlp = CEFN(network_depth=network_depth, dim=dim, hidden_features=int(mlp_ratio) * dim, out_features=dim)

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.Linear1(x)
        x = self.DWConv(x)
        x = self.DWDConv(x)
        x = self.Linear2(x) + identity

        identity = x
        x = self.norm2(x)
        x = self.cemlp(x) + identity
        return x

# BasicLayer name is used there
class LKDBlocks(nn.Module):
    def __init__(self, network_depth, dim, depth, mlp_ratio=4.):
        #self, network_depth, dim, depth, num_heads, mlp_ratio=4.,
		#norm_layer=nn.LayerNorm, window_size=8,
		#attn_ratio=0., attn_loc='last', conv_type=None
        #
        super().__init__()
        
        self.dim = dim
        self.depth = depth


#----------------------------------These Lines are extra used there-------------------------------
        # This line is missing
        #attn_depth = attn_ratio * depth

        #if attn_loc == 'last':
		#	use_attns = [i >= depth-attn_depth for i in range(depth)]
		#elif attn_loc == 'first':
		#	use_attns = [i < attn_depth for i in range(depth)]
		#elif attn_loc == 'middle':
		#	use_attns = [i >= (depth-attn_depth)//2 and i < (depth+attn_depth)//2 for i in range(depth)]





        # build blocks
        self.blocks = nn.ModuleList([
            # it name was TransformerBlock
            LKDBlock(network_depth=network_depth,
                     dim=dim,
                     mlp_ratio=mlp_ratio
                     
                     #These lines are not included
                     #num_heads=num_heads,
                     #norm_layer=norm_layer,
                     #window_size=window_size,
                     #shift_size=0 if (i % 2 == 0) else window_size // 2,
                     #use_attn=use_attns[i], conv_type=conv_type
                     )
            for i in range(depth)])

    # This model is same
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

#-------------------------------- Same Method used------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x

#----------------------------------------------------------------------------------------------

#---------------------- Same Method Used-------------------------------------------------------
class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x
#----------------------------------------------------------------------------------------------

#------------------------------SAme Method used-------------------------------------------------
class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out
#----------------------------------------------------------------------------------------------


##_____________________________Checked this moodel, .----------
class Dehaze(nn.Module):
    def __init__(self, in_chans=3, out_chans=4,  # window_size=8, is missing here
                 # --------Used same three line, however, there were many others.
                 embed_dims=[24, 48, 96, 48, 24],
                 mlp_ratios=[2., 4., 4., 2., 2.],
                 num_heads=[2, 4, 6, 1, 1],
				 #attn_ratio=[1/4, 1/2, 3/4, 0, 0],
				 #conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
                 #norm_layer=[RLN, RLN, RLN, RLN, RLN]):
                 depths=[16, 16, 16, 8, 8], ):
        super(Dehaze, self).__init__()

        # This comment line is missing only
        self.patch_size = 4
        #self.window_size = window_size
        self.mlp_ratios = mlp_ratios

        # This line is same
        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        # backbone
        # Few Lines are changed, THis layer is changed 
        self.layer1 = LKDBlocks(network_depth=sum(depths), dim=embed_dims[0], depth=depths[0], 
                                mlp_ratio=mlp_ratios[0])

        # Same Lines
        # Few Lines are changed
        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])

        # Same Lines
        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        # Same Lines, THis layer is changed
        self.layer2 = LKDBlocks(network_depth=sum(depths), dim=embed_dims[1],
                                 depth=depths[1], mlp_ratio=mlp_ratios[1])

        # This line is same
        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        #This line is same
        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        # Same Lines, THis layer is changed
        self.layer3 = LKDBlocks(network_depth=sum(depths), dim=embed_dims[2], depth=depths[2], 
                                mlp_ratio=mlp_ratios[2])

        # This line is same
        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        # These lines are same
        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SKFusion(embed_dims[3])

        # Same Lines, THis layer is changed
        self.layer4 = LKDBlocks(network_depth=sum(depths), dim=embed_dims[3], depth=depths[3], 
                                mlp_ratio=mlp_ratios[3])

        # This line is same
        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        # These both lines are same
        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = SKFusion(embed_dims[4])

        # Same Lines, THis layer is changed
        self.layer5 = LKDBlocks(network_depth=sum(depths), dim=embed_dims[4], depth=depths[4], 
                                mlp_ratio=mlp_ratios[4])

        # Same Line
        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        #This line is extra here
        patch_size = self.patch_size

        mod_pad_h = (patch_size - h % patch_size) % patch_size
        mod_pad_w = (patch_size - w % patch_size) % patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    
    # all lines are included
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.layer1(x)
        skip1 = x

        x = self.patch_merge1(x)
        x = self.layer2(x)
        skip2 = x

        x = self.patch_merge2(x)
        x = self.layer3(x)
        x = self.patch_split1(x)

        x = self.fusion1([x, self.skip2(skip2)]) + x
        x = self.layer4(x)
        x = self.patch_split2(x)

        x = self.fusion2([x, self.skip1(skip1)]) + x
        x = self.layer5(x)
        x = self.patch_unembed(x)
        return x

    # all lines are included
    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        feat = self.forward_features(x)
        K, B = torch.split(feat, (1, 3), dim=1)

        x = K * x - B + x
        x = x[:, :, :H, :W]
        return x

#-----------------------------------------------------------------------------------------------


#------------------------------ Used this method with other name. But, few lines are used.-----
def LKD_t():
    return Dehaze(
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[4., 4., 4., 4., 4.],
        depths=[1, 1, 2, 1, 1],
        # These lines are not used
        #num_heads=[2, 4, 6, 1, 1],
		#attn_ratio=[1/4, 1/2, 3/4, 0, 0],
		#conv_type=['Conv', 'Conv', 'Conv', 'Conv', 'Conv'])
    )


#------------------------------ Used this method with other name. But, few lines are used.-----
def LKD_s():
    return Dehaze(
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[4., 4., 4., 4., 4.],
        depths=[2, 2, 4, 2, 2],
        # These lines are not used
        #num_heads=[2, 4, 6, 1, 1],
		#attn_ratio=[1/4, 1/2, 3/4, 0, 0],
		#conv_type=['Conv', 'Conv', 'Conv', 'Conv', 'Conv'])
    )


#------------------------------ Used this method with other name. But, few lines are used.-----
def LKD_b():
    return Dehaze(
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[4., 4., 4., 4., 4.],
        depths=[4, 4, 8, 4, 4],
        # These lines are not used
        #num_heads=[2, 4, 6, 1, 1],
		#attn_ratio=[1/4, 1/2, 3/4, 0, 0],
		#conv_type=['Conv', 'Conv', 'Conv', 'Conv', 'Conv'])
    )


#------------------------------ Used this method with other name. But, few lines are used.-----
def LKD_l():
    return Dehaze(
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[4., 4., 4., 4., 4.],
        depths=[8, 8, 16, 8, 8],
        # These lines are not used
        #num_heads=[2, 4, 6, 1, 1],
		#attn_ratio=[1/4, 1/2, 3/4, 0, 0],
		#conv_type=['Conv', 'Conv', 'Conv', 'Conv', 'Conv'])
    )
