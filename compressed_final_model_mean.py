import torch
import torch.optim as optim
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os

try:
    import bm3d
except ImportError:  # BM3D is optional for configurations that do not use filtering.
    bm3d = None
os.environ["PYTORCH_NO_SHM"] = "1"


def psnr(img1, img2, max_val=1.0):
    """计算PSNR"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 10 * torch.log10(max_val ** 2 / mse)

def intensity_get(img_tensor):
    """获取亮度通道（简单实现）"""
    # img_tensor: [H, W, 3]
    brightness = 0.299 * img_tensor[:, :, 0] + \
                 0.587 * img_tensor[:, :, 1] + \
                 0.114 * img_tensor[:, :, 2]
    return brightness.unsqueeze(0).unsqueeze(0)  # 模拟原代码的维度输出

def set_random_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# ====================== 核心模型与类定义 ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GammaMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x, y):
        feat = torch.stack([x, y], dim=-1)
        h = F.relu(self.fc1(feat))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

class LogExpEnhancement:
    def __init__(
        self,
        image_path,
        gt_path=None,
        exp_path=None,
        random_number=6272,
        filter=True,
        quality='NORMAL',
        c=1,
        param_range=(0.3, 1.0),
        residual=None,
        mode='auto',
        overexposure_threshold=0.7,
        residual_threshold=0.1,
        bm3d_sigma=20.0,
    ):
        """
        参数:
            image_path: 输入低光图像路径
            gt_path: 真值图像路径（可选，仅用于评估，无真值可传None）
            exp_path: 输出路径（可选）
        """
        # 读取输入图像
        self.img = Image.open(image_path).convert('RGB')
        self.original_array = np.array(self.img, dtype=np.float32)
        self.img_tensor = torch.tensor(self.original_array / 255.0, dtype=torch.float32).to(device)
        
        # 读取真值图像（如果提供）
        self.gt_tensor = None
        if gt_path is not None:
            self.gt_img = Image.open(gt_path).convert('RGB')
            self.gt_array = np.array(self.gt_img, dtype=np.float32)
            self.gt_tensor = torch.tensor(self.gt_array / 255.0, dtype=torch.float32).to(device)
            self.gt_brightness = self.calculate_brightness(self.gt_tensor)
        else:
            self.gt_brightness = torch.tensor(0.0)

        self.input_brightness = self.calculate_brightness(self.img_tensor)
        self.height, self.width, self.channels = self.img_tensor.shape
        if mode not in {'auto', 'brighten', 'darken'}:
            raise ValueError("mode must be one of: auto, brighten, darken")
        self.residual = False
        self.residual_setting = residual
        self.mode = mode
        self.overexposure_threshold = overexposure_threshold
        self.residual_threshold = residual_threshold
        self.bm3d_sigma = bm3d_sigma
        self.objective = 1 
        # 初始化配置
        set_random_seed(random_number)
        self.quality = quality
        self.filter = filter
        self.c = c
        self.param_range = param_range
        self.exp_path = exp_path
      
        # 初始化模型
        self.gamma_mlp = GammaMLP().to(device)
        self.reference_intensity = intensity_get(self.img_tensor).squeeze(0).permute(1, 2, 0)
        self.empty_tensor = torch.zeros((self.height, self.width, 3), device=device)
        
        # 生成坐标网格
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, self.height, device=device),
            torch.linspace(-1, 1, self.width, device=device),
            indexing='ij'
        )
        self.coord_x = xs
        self.coord_y = ys

        # 用于保存中间状态的变量
        self.saved_mlp_state = None  # 15次优化后的MLP参数
        self.brightness_15 = None     # 15次优化后的亮度
        self.brightness_25 = None     # 25次优化后的亮度

    def apply_log_transform(self, img_tensor):
        return torch.log(1e-10 + img_tensor)

    def apply_exp_transform(self, log_tensor):
        return torch.exp(log_tensor) 

    def scale_params(self):
        log_gamma = self.gamma_mlp(self.coord_x, self.coord_y)
        gamma_min, gamma_max = self.param_range
        gamma = gamma_min + (gamma_max - gamma_min) * torch.sigmoid(log_gamma)
        #gamma = log_gamma
        return gamma

    def forward(self, gamma):
        log_transformed = self.apply_log_transform(self.img_tensor)
        param_applied = log_transformed * gamma
        if self.residual:
            exp_transformed = self.apply_exp_transform(param_applied) + self.apply_exp_transform(log_transformed)    
        else:
            exp_transformed = self.apply_exp_transform(param_applied) 
        return exp_transformed   

    def calculate_brightness(self, img_tensor):
        brightness = 0.299 * img_tensor[:, :, 0] + \
                     0.587 * img_tensor[:, :, 1] + \
                     0.114 * img_tensor[:, :, 2]
        return torch.mean(brightness)

    def optimize_brightness(self,num_iterations,learning_rate=0.1):
        """执行25次优化，在第15次暂存状态"""
        optimizer = optim.Adam(self.gamma_mlp.parameters(), lr=learning_rate)
        print(f"Input Brightness: {self.input_brightness.item():.5f}")
        if self.gt_tensor is not None:
            print(f"GT Brightness:    {self.gt_brightness.item():.5f}")
        
        brightness_history = []
        num_iterations = num_iterations
        input_brightness = self.input_brightness.item()
        if self.mode == 'auto':
            correction_mode = (
                'darken'
                if input_brightness >= self.overexposure_threshold
                else 'brighten'
            )
            if correction_mode == 'darken':
                self.param_range = (3.0, 3.5)
        else:
            correction_mode = self.mode

        self.objective = -1 if correction_mode == 'darken' else 1
        if self.residual_setting is None:
            self.residual = (
                correction_mode == 'brighten'
                and input_brightness < self.residual_threshold
            )
        else:
            self.residual = bool(self.residual_setting)
        
        #self.residual = True #this implementation is for ablation
        for i in range(num_iterations):
            optimizer.zero_grad()
            
            scaled_params = self.scale_params()
            enhanced_img = self.forward(scaled_params)
            brightness = self.calculate_brightness(enhanced_img)
            
            loss = -1*self.objective*brightness 
            loss.backward()
            optimizer.step()
            
            brightness_history.append(brightness.item())
            '''
            # 第15次迭代暂存
            if i == 14: 
                self.saved_mlp_state = self.gamma_mlp.state_dict().copy()
                self.brightness_15 = brightness.item()
                print(f"[Iter 15/25] Saved. Brightness: {self.brightness_15:.5f}")

            # 第25次迭代完成
            if i == num_iterations-1:
                self.brightness_25 = brightness.item()
                print(f"[Iter 25/25] Finished. Brightness: {self.brightness_25:.5f}")
            '''
        # 计算初始PSNR（仅用于参考）
        with torch.no_grad():
            gamma = self.scale_params()
            enhanced_img = self.forward(gamma)
            psnr_value_zero = psnr(enhanced_img, self.empty_tensor, max_val=1.0)

        return brightness_history, self.input_brightness.item(),self.gt_brightness.item(),

    def get_enhanced_image(self):
        """Generate the enhanced RGB image with optional BM3D filtering."""
        with torch.no_grad():
            gamma_15 = self.scale_params()
            enhanced_img = self.forward(gamma_15)
            enhanced_img = torch.clamp(enhanced_img, 0.0, 1.0)
            enhanced_img_cpu = enhanced_img.cpu()
            enhanced_img_np = (enhanced_img_cpu * 255).clamp(0, 255).byte().numpy()
            
            if self.filter:
                if bm3d is None:
                    raise ImportError(
                        "BM3D filtering was requested, but the 'bm3d' package is not "
                        "installed. Run: pip install bm3d"
                    )
                sigma = self.bm3d_sigma
                print(f"Applying BM3D denoising (sigma={sigma})...")
                denoised_rgb = bm3d.bm3d_rgb(
                    z=enhanced_img_np.astype(np.float32),
                    sigma_psd=sigma,
                )
                enhanced_img_np = np.clip(denoised_rgb, 0, 255).astype(np.uint8)
            
            return Image.fromarray(enhanced_img_np, 'RGB')
    def get_enhanced_intensity(self):
        with torch.no_grad():
            gamma = self.scale_params()
            enhanced_img = self.forward(gamma)
            intensity = intensity_get(enhanced_img).squeeze(0).permute(1,2,0) #max_values
            intensity_cpu = intensity[:,:,0].cpu()
            intensity_np = (intensity_cpu * 255).clamp(0, 255).byte().numpy()
            return Image.fromarray(intensity_np,'L') 
    def get_enhanced_image_old(self):
        """获取增强后的图像"""
        with torch.no_grad():
            gamma = self.scale_params()
            enhanced_img = self.forward(gamma)
            
            #enhanced_img = bm3d_denoise_rgb(enhanced_img,20.0/255.0)
                       # 将张量移回CPU并进行后处      
            enhanced_img_cpu = enhanced_img.cpu()
            enhanced_img = (enhanced_img_cpu * 255).clamp(0, 255).byte().numpy()
            
            if self.filter:
                sigma = 20   #all other bm3d value 20
                # 3. 调用 bm3d 滤波（彩色图用 bm3d.bm3d_rgb）
                denoised_rgb = bm3d.bm3d_rgb(
                z=enhanced_img.astype(np.float32),
                sigma_psd=sigma,
)
                # 4. 结果处理（转回 uint8 并裁剪，再转为 BGR 用于保存）
                enhanced_img = np.clip(denoised_rgb, 0, 255).astype(np.uint8)
            return Image.fromarray(enhanced_img, 'RGB')
    def get_reference_image(self):
        with torch.no_grad():
            reference_img = self.empty_tensor 
            reference_img_cpu = reference_img.cpu()
            reference_img = (reference_img_cpu * 255).clamp(0, 255).byte().numpy()
            return Image.fromarray(reference_img, 'RGB')    
    def get_difference_image(self):
        """获取增强后的图像"""
        with torch.no_grad():
            #enhanced_img,brightness = self.forward()
            enhanced_img = self.enhance_ratio*self.forward()
            enhanced_img = torch.relu(0.13-(enhanced_img - self.img_tensor))
            #enhanced_img = self.gaussian(enhanced_img) #not sure whether the code is accurate 
            # 将张量移回CPU并进行后处理
            enhanced_img_cpu = enhanced_img.cpu()
            enhanced_img = (enhanced_img_cpu * 255).clamp(0, 255).byte().numpy()
            return Image.fromarray(enhanced_img, 'RGB')    
    def get_params(self):
        with torch.no_grad():
            params = self.scale_params()
            mean = torch.mean(params)
            print(f"param_mean:{torch.mean(params)}".format())
            params = params - mean
            params_cpu = params[:,:,0].cpu()
            params = (params_cpu * 255).clamp(0, 255).byte().numpy()
            #print(params.shape)
            return Image.fromarray(params,'L')
    def get_increment(self):
        with torch.no_grad():
            if self.quality == 'BAD':
                increment_img_cpu = self.increment_rgb.cpu()
                increment_img = (increment_img_cpu * 255).clamp(0, 255).byte().numpy()
                #print(increment_img.max())
                return Image.fromarray(increment_img, 'RGB')
    def get_intensity(self):
        with torch.no_grad():
            intensity = self.reference_intensity#*self.intensity_weight
            intensity_cpu = intensity[:,:,0].cpu()
            intensity_np = (intensity_cpu * 255).clamp(0, 255).byte().numpy()
            return Image.fromarray(intensity_np,'L') 
        
    def get_parameter_matrix(self):
        """获取参数矩阵"""
        with torch.no_grad():
            return self.scale_params().cpu().numpy()
    def get_original_brightness(self):
        """计算原始图像亮度"""
        # 将原始数组转换为张量并移到设备
        original_tensor = torch.tensor(self.original_array / 255.0, dtype=torch.float32).to(device)
        original_brightness = 0.299 * original_tensor[:, :, 0] + \
                             0.587 * original_tensor[:, :, 1] + \
                             0.114 * original_tensor[:, :, 2]
        return torch.mean(original_brightness).item()
    def get_original_color(self):
        """计算原始图像亮度"""
        # 将原始数组转换为张量并移到设备
        original_tensor = torch.tensor(self.original_array / 255.0, dtype=torch.float32).to(device)
        R_G_B_summation = self.calculate_rgbsummation(original_tensor)
        color = torch.mean(self.rgb_channel_mask*original_tensor/R_G_B_summation)
        return torch.mean(color).item()
