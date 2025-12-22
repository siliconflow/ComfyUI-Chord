import torch
import torch.nn.functional as F
from torch.signal.windows import hann
import torch.fft as fft_module

def compute_divergence(fx: torch.tensor, fy: torch.tensor) -> torch.tensor:
    div_x = F.pad(fx[:, :, 1:] - fx[:, :, :-1], (0, 1, 0, 0), mode='constant')
    div_y = F.pad(fy[:, 1:, :] - fy[:, :-1, :], (0, 0, 0, 1), mode='constant')
    return div_x + div_y

def solve_poisson_fft(div: torch.tensor, h: int, w: int) -> torch.tensor:
    fft_div = fft_module.fft2(div)

    kx = fft_module.fftfreq(2*w, device=div.device) * 2 * torch.pi
    ky = fft_module.fftfreq(2*h, device=div.device) * 2 * torch.pi
    kx, ky = torch.meshgrid(kx, ky, indexing='xy')
    epsilon = 1e-9
    denom = 4 - 2 * torch.cos(kx) - 2 * torch.cos(ky)
    denom = torch.where(torch.abs(denom) > epsilon, denom, epsilon) 

    height_map_full = torch.real(fft_module.ifft2(fft_div / denom))
    return torch.nan_to_num(height_map_full[:, :h, :w])
    
def apply_window_function(gradient: torch.tensor) -> torch.tensor:
    hann_window = hann(gradient.shape[-2], device=gradient.device)[:, None] * hann(gradient.shape[-1], device=gradient.device)[None, :]
    return gradient * hann_window
    
def compute_height(normal_map: torch.tensor, epsilon: float = 1e-8) -> torch.tensor:
    h, w = normal_map.shape[-2:]      
    nz = normal_map[:, 2]
    nz_safe = torch.where(torch.abs(nz) > epsilon, nz, epsilon)
    fx = normal_map[:, 0] / nz_safe
    fy = normal_map[:, 1] / nz_safe

    fx = apply_window_function(fx)
    fy = apply_window_function(fy)
    
    div = compute_divergence(fx, fy)
    div = F.pad(div, (0, w, 0, h), mode='constant')
    height_map = solve_poisson_fft(div, h, w)
    return height_map - torch.mean(height_map)

def extend_normal_map(normal_map: torch.tensor, region_size: int) -> torch.tensor:
    larger_normal_map = F.pad(normal_map, (region_size, region_size, region_size, region_size), mode='circular')
    return larger_normal_map

def define_subregions(
    h: int,
    w: int,
    min_region_size: int=128,
    overlap_factor: float = 0.5
) -> list:
    step_size = int(min_region_size - min_region_size * overlap_factor)
    if step_size <= 0:
        step_size = min_region_size
    overlap_size = int(min_region_size * overlap_factor)
    subregions = []
    for y in range(0, h, step_size):
        for x in range(0, w, step_size):
            y_end = min(y + min_region_size + overlap_size, h)
            x_end = min(x + min_region_size + overlap_size, w)
            subregions.append((y, y_end, x, x_end))
    return subregions

def cosine_smoothing(x: torch.tensor) -> torch.tensor:
    return 0.5 * (1 - torch.cos(torch.pi * x))

def create_subregions(
    normal_map: torch.tensor,
    subregions: list,
) -> list:
    results = []
    for region in subregions:
        y, y_end, x, x_end = region
        sub_map = normal_map[:, y:y_end, x:x_end]
        results.append(sub_map)
    return results

def cosine_blending_update_weight(
    sub_weight_map: torch.tensor,
    subregion,
    shape,
    region_size: int = 128,
) -> None:
    h, w = sub_weight_map.shape[-2:]
    d = sub_weight_map.device
    y_start, y_end, x_start, x_end = subregion
    if y_start > 0:
        overlap = min(region_size, h)
        y_smooth = cosine_smoothing(torch.linspace(0, 1, overlap, device=d))[:, None]
        sub_weight_map[:overlap, :] *= y_smooth
    if y_end < shape[0]:
        overlap = min(region_size, h)
        y_smooth = cosine_smoothing(torch.linspace(1, 0, overlap, device=d))[:, None]
        sub_weight_map[-overlap:, :] *= y_smooth
    if x_start > 0:
        overlap = min(region_size, w)
        x_smooth = cosine_smoothing(torch.linspace(0, 1, overlap, device=d))
        sub_weight_map[:, :overlap] *= x_smooth
    if x_end < shape[1]:
        overlap = min(region_size, w)
        x_smooth = cosine_smoothing(torch.linspace(1, 0, overlap, device=d))
        sub_weight_map[:, -overlap:] *= x_smooth

def combine_sub_height_maps(
    height_maps: list,
    subregions: list,
    h: int,
    w: int,
    epsilon: float = 1e-8,
) -> torch.tensor:
    height_map = torch.zeros((h, w), device=height_maps[0].device)
    weight_map = torch.zeros((h, w), device=height_maps[0].device)

    for subregion, sub_height_map in zip(subregions, height_maps):
        y_start, y_end, x_start, x_end = subregion
        y_slice = slice(y_start, y_end)
        x_slice = slice(x_start, x_end)
        
        sub_weight_map = torch.ones_like(sub_height_map)
        cosine_blending_update_weight(
            sub_weight_map,
            subregion,
            (h, w)
        )

        height_map[y_slice, x_slice] += sub_height_map * sub_weight_map
        weight_map[y_slice, x_slice] += sub_weight_map

    return height_map / (weight_map + epsilon)

def crop_height_map(height_map: torch.tensor, original_shape: tuple, region_size: int=128) -> torch.tensor:
    original_height, original_width = original_shape
    return height_map[
        region_size:region_size + original_height,
        region_size:region_size + original_width
    ]

def normalize_height_map(data: torch.tensor, eps: float = 1e-8) -> torch.tensor:
    return (data - data.min()) / (data.max() - data.min() + eps)

def normal_to_height(
        normal_map: torch.tensor,
        subdivisions: int = 16,
        min_region_size: int = 128,
        skip_normalize_normal: bool = False,
) -> torch.tensor:
    '''
    Args:
        normal_map: torch.tensor(1, 3, H, W)
        subdivisions: int, subdivision level at each edge
        min_region_size: int, minimal region size
        skip_normalize_normal: bool, if skip normalization of input normal map
    '''
    if normal_map.dim() == 4:
        try: assert normal_map.shape[0] == 1
        except: breakpoint()
        normal_map = normal_map.squeeze(0)
    h, w = normal_map.shape[-2:]
    if not skip_normalize_normal: 
        normal_map = F.normalize(normal_map * 2.0 - 1.0, dim=0)
    region_size = min(
        max(min(h, w) // subdivisions, min_region_size),
        min(h, w)
    )
    larger_normal_map = extend_normal_map(normal_map, region_size)
    lh, lw = larger_normal_map.shape[-2:]
    subregions = define_subregions(lh, lw, region_size)
    map_batch = create_subregions(larger_normal_map, subregions)
    sub_height_maps = []
    for m in map_batch:
        sub_height_maps.append(compute_height(m.unsqueeze(0)).squeeze(0))
    height_combined = combine_sub_height_maps(sub_height_maps, subregions, lh, lw)
    height_cropped = crop_height_map(height_combined, (h, w), region_size)
    return normalize_height_map(height_cropped)
