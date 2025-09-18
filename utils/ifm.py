"""
Copyright (c) 2025 Vasiliki Ismiroglou. All rights reserved.
This code is part of the paper: "Sea-ing Through Scattered Rays: Revisiting the Image Formation Model for
Realistic Underwater Image Generation" arXiv:
"""
import pandas as pd
import cv2
import os
import numpy as np

from utils.utils import get_sr, get_coeffs, get_updated_coeffs
from utils.depth import calc_depth
import utils.gaussian_random_fields as grf

#TODO: watertype parser. A system were the user can either pass a list of water type names or pass coefficients directly.
#TODO: fix variable names inhomog_map vs turbidity, d vs vdepth, image vs img
class IFM():
    def __init__(self,
                out_dir:str,
                wtypes:list[str],
                vdepths:list[float|int],
                fscatter:bool,
                g:float,
                mu:float,
                inhomog:bool,
                grf_min:float,
                grf_max:float,
                camera=None
                ):
        """
        Initializes the IFM class.

        Args:
            out_dir (str): Directory path for output files.
            wtypes (list[str]): List of water types.
            vdepths (list[float | int]): List of vertical depths.
            fscatter (bool): Flag to enable or disable forward scattering.
            g (float): Asymmetry parameter for effective scattering. Indicates how much light is scattered away from the camera.
            mu (float): Asymmetry parameter for effective scattering. Indicates how much backscattered light is scattered towards the camera.
            inhomog (bool): Flag indicating if the medium is inhomogeneous.
            grf_min (float): Minimum value for the Gaussian random field.
            grf_max (float): Maximum value for the Gaussian random field.
            camera (optional): Camera configuration object or None.

        Side Effects:
            Creates the output directory if it does not exist.
            Initializes internal state variables.
        """
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.wtypes = wtypes
        self.vdepths = vdepths
        self.fscatter = fscatter
        self.g = g
        self.mu = mu
        self.inhomog = inhomog
        self.grf_min = grf_min
        self.grf_max = grf_max

        self.sr, self.full = get_sr(camera)


    def _get_scaled_grf(self, img):
        inhomog_map = grf.gaussian_random_field(alpha=4, size=max(img.shape))[:img.shape[0], :img.shape[1]]
        t_min, t_max = inhomog_map.min(), inhomog_map.max()
        turb_norm = (inhomog_map - t_min) / (t_max - t_min)

        turb_rescaled = turb_norm * (self.grf_max - self.grf_min) + self.grf_min

        mean_val = np.mean(turb_rescaled)
        correction = 1.0 / mean_val
        turb_final = turb_rescaled * correction

        return turb_final

    def _get_direct_trans(self, img:np.ndarray, z:np.ndarray, beta:np.ndarray, K:np.ndarray, d:float) -> np.ndarray:
        """
        Calculate the direct transmission term with full coefficients, considering downwelling attenuation at depth d. 

        Args:
            img (np.ndarray): Input image of shape (H, W, C).
            z (np.ndarray): 2D array representing the horizontal depth map of the scene (H x W).
            beta (np.ndarray): Beam attenuation coefficients for each channel.
            K (np.ndarray): Downwelling attenuation coefficients for each channel.
            d (float): Vertical depth.

        Returns:
            np.ndarray: The direct transmission component.
        """
        degraded_img = img * np.exp(-z[:, :, np.newaxis] * beta) * np.exp(-K * d)
        return degraded_img

    def _get_forward_scatter(self, img:np.ndarray, z:np.ndarray, beta:np.ndarray, G:np.ndarray, phi:float, K:np.ndarray, d:float) -> np.ndarray:
        """
        Calculate the forward scattering term.

        Args:
            img (np.ndarray): Input image of shape (H, W, C).
            z (np.ndarray): 2D array representing the horizontal depth map of the scene (H x W).
            beta (np.ndarray): Beam attenuation coefficients for each channel.
            G (np.ndarray): Effective/apparent attenuation coefficients for each channel.
            phi (float): Blurring factor
            K (np.ndarray): Downwelling attenuation coefficients for each channel.
            d (float): Vertical depth.

        Returns:
            np.ndarray: The forward scattering component.
        """

        def depth_blur_in_bins(weighted_image, num_bins=30):
            """
            image: RGB image as a NumPy array (H x W x 3)
            depth_map: depth image (grayscale or 1-channel float) (H x W)
            num_bins: how many depth intervals to use
            max_blur: max kernel size for the farthest depth bin (must be odd)
            """
            # Compute bin edges
            bins = np.linspace(0, np.max(z), num_bins + 1)
            blurred_layers = []
        
            # Precompute blurred images for each bin
            for i in range(num_bins):
                # Determine blur amount for this bin
                blur_strength = phi * (bins[i] + bins[i + 1]) / 2 
                blurred = cv2.GaussianBlur(weighted_image, (15, 15), blur_strength)

                blurred_layers.append(blurred)

            # Create output by assigning pixels based on depth bin
            output = np.zeros_like(weighted_image)
            for i in range(num_bins):
                mask = (z >= bins[i]) & (z < bins[i + 1])
                for c in range(3):  # apply to each channel
                    output[:, :, c][mask] = blurred_layers[i][:, :, c][mask]

            return output
        
        def depth_based_blur(weighted_image, phi, kernel='lorentzian'):
            def gaussian_kernel(size, sigma):
                ax = np.linspace(-(size // 2), size // 2, size)
                xx, yy = np.meshgrid(ax, ax)
                kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
                kernel /= np.sum(kernel)
                return kernel

            def lorentzian_kernel(size, sigma):
                ax = np.linspace(-(size // 2), size // 2, size)
                xx, yy = np.meshgrid(ax, ax)
                rr_sq = xx**2 + yy**2
                kernel = sigma / (2 * np.pi * (sigma**2 + rr_sq)**(1.5))
                kernel /= np.sum(kernel)
                return kernel
            
            def pixelwise_blur(img, depth, phi=1.0, kernel_size=15, kernel='gaussian'):
                if kernel == 'gaussian':
                    kernel_func = gaussian_kernel
                elif kernel == 'lorentzian':
                    kernel_func = lorentzian_kernel
                else:
                    raise ValueError("Unsupported kernel type. Use 'gaussian' or 'lorentzian'.")
                
                h, w = img.shape[:2]
                pad = kernel_size // 2
                image_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
                output = np.zeros_like(img, dtype=np.float32)

                for y in range(h):
                    for x in range(w):
                        sigma = phi * depth[y, x]
                        kernel = kernel_func(kernel_size, sigma)
                        patch = image_padded[y:y+kernel_size, x:x+kernel_size]
                        output[y, x] = np.sum(patch * kernel[..., None], axis=(0,1)) if img.ndim == 3 else np.sum(patch * kernel)

                return output

            kernel_size = 15
            blurred_img = pixelwise_blur(weighted_image, z, phi, kernel_size=int(kernel_size), kernel=kernel)
            return blurred_img

        # Perform scaling    
        forward_scatter = img * (np.exp(-z[:, :, np.newaxis] * G) - np.exp(-z[:, :, np.newaxis] * beta)) * np.exp(-K * d)

        # Calculate blur
        if phi is not None:
            # forward_scatter = depth_based_blur(forward_scatter, phi)
            forward_scatter = depth_blur_in_bins(forward_scatter, num_bins=30)

        return forward_scatter

    def _get_backscatter(self, z:np.ndarray, beta:np.ndarray, b:np.ndarray, K:np.ndarray, d:float, E=np.array([1, 1, 1])) -> np.ndarray:
        """
        Calculate the backscatter component optionally using camera-specific spectral response.

        Args:
            z (np.ndarray): 2D array representing the horizontal depth map of the scene (H x W).
            beta (np.ndarray): Beam attenuation coefficients for each wavelength or channel.
            b (np.ndarray): Scattering coefficients for each wavelength or channel.
            K (np.ndarray): Downwelling attenuation coefficients for each wavelength or channel.
            d (float): Vertical depth.
            sr (pd.DataFrame or None): Spectral response data. If a DataFrame, considers camera-specific spectral response. Otherwise, uses inherent coefficients.
            E (np.ndarray): Ambient light at the surface.

        Returns:
            np.ndarray: The backscatter component of the image, with shape (H, W, C)
        """
        if isinstance(self.sr, pd.DataFrame):
            height, width = z.shape
            backscatter = np.zeros((height, width, 3), dtype=np.float32)
            backlight = np.zeros(3, dtype=np.float32)

            channels = ['R', 'G', 'B']

            for i, ch in enumerate(channels):
                ch_backscatter = 0
                for nm in range(400, 701, 50): # This is hardcoded because it is the step at which the camera database and coefficient database are aligned.
                    wv_sr = self.sr[self.sr.iloc[:, 1] == ch][str(nm)].values[0] # This is the spectral response for this channel and this wavelength
                    index = int((nm - 400) / 50)

                    wv_backlight = b[index] * E[i] * np.exp(-K[index] * d)/ beta[index]
                    
                    # This is the backscatter contribution for this channel and this wavelength. It includes beam attenuation.
                    ch_backscatter += wv_sr * wv_backlight * (1 - np.exp(-z * beta[index])) 

                backscatter[:, :, i] = ch_backscatter
        else:
            print('Calculating backscatter using inherent coefficients.')
            backlight = b * E * np.exp(- K * d)/ beta
            backscatter = backlight * (1 - np.exp(-z[:, :, np.newaxis] * beta))

        return backscatter   

    def _process_coeffs(self, wtype, z):
        """
        Processes and updates coefficient arrays based on the specified window type and input parameter `z`.
        Parameters
        ----------
        wtype : str
            Water type
        z : array-like
            Horizontal depth
        Returns
        -------
        a : float
            Absorption coefficient IOP
        b : float
            Scattering coefficient IOP
        beta : float
            Attenuation coefficient (a + b) IOP
        K : float
            Downwelling attenuation coefficient IOP
        phi : float
            Blurring factor
        G : float
            Effective attenuation coefficient (a + g * b) IOP
        beta_direct : float
            Attenuation coefficient accounting for camera spectral response AOP
        G_direct:
            Effective attenuation coefficient accounting for camera spectral response AOP
        -----
        If a camera type is not provided, beta = beta_direct and G = G_direct.
        If forward scattering is not considered, phi, G and G_direct get default values and are not used.
        """

        a, b, beta, K = get_coeffs(wtype, full=self.full)
        phi = 0.3 * np.mean(b)
        # phi = simple_eval(phi_exp, functions={'mean':np.mean}, names={'a':a, 'b':b, 'beta':beta, 'K':K, 'g':g, 'mu':mu}) if isinstance(phi_exp, str) else phi_exp
        G = a + self.g * b # Effective coefficient

        if isinstance(self.sr, pd.DataFrame):
            beta_direct = get_updated_coeffs(z, beta, self.sr)
            G_direct = get_updated_coeffs(z, G, self.sr)
        else:
            beta_direct = beta
            G_direct = G
        
        return a, b, beta, K, phi, G, beta_direct, G_direct

    def _generate_single_image(self, image, z, inhomog_map, vdepth, wtype):
        print('Processing for water type:', wtype)
        a, b, beta, K, phi, G, beta_direct, G_direct = self._process_coeffs(wtype, z)

        if self.fscatter:
            # Notice the replacement of beta and b with apparent coefficients.
            # The blur here plays two roles. Firstly, the forward scattering equivalent for the backscatter term, would mean there is some 
            # blurring and the backscatter doesn't fully capture the scene geometry. Secondly, the depth maps generated by 
            # monocular depth estimators have sharp edges which can cause artifacts.
            z_blurred = cv2.GaussianBlur(z, (21, 21), 60)
            backscatter = self._get_backscatter(z=z_blurred*inhomog_map, beta=G, b=self.mu*b, K=K, d=vdepth) #default is 0.3  #cv2.GaussianBlur(h_depth, (int(height/8)*2+1,int(width/8)*2+1),20)*unhomog
        else:
            z_blurred = cv2.GaussianBlur(z, (21, 21), 60)
            backscatter = self._get_backscatter(z=z_blurred*inhomog_map, beta=beta, b=b, K=K, d=vdepth)

        # If the full downwelling coefficients are provided, after this point only the CH equivalent are used. This can be updated in the future.
        # This is hardcoded and only works if the coefficient .csv file follows the same structure as the one indicated by this code.
        if K.shape != (3,):
            K = K[[4,3,1]]
        
        direct = self._get_direct_trans(img=image, z=z*inhomog_map, beta=beta_direct, K=K, d=vdepth)

        if self.fscatter:
            forward = self._get_forward_scatter(img=image, z=z*inhomog_map, beta=beta_direct, G=G_direct, phi=phi, K=K, d=vdepth)
            degraded_image = np.clip((direct + forward + backscatter), 0, 1) #
        else:
            degraded_image = np.clip((direct + backscatter), 0, 1)
        
        return degraded_image

    def generate(self, image, z_min, z_max, gamma):
        z, image = calc_depth(image, type='relative', min=z_min, max=z_max, gamma=gamma)
        
        if self.inhomog:
            inhomog_map = self._get_scaled_grf(image)
        else:
            inhomog_map = 1
        
        for vdepth in self.vdepths:
            print(f'Setting vertical depth to {vdepth}m')
            for wtype in self.wtypes:
                degraded_image = self._generate_single_image(image, z, inhomog_map, vdepth, wtype)
                degraded_image = cv2.cvtColor((degraded_image*255).astype('uint8'), cv2.COLOR_RGB2BGR)
                fname = f'{wtype}_{vdepth}_{int(self.fscatter)}_{int(self.inhomog)}_{self.g}_{self.mu}_{self.grf_min}_{self.grf_max}_{z_min}_{z_max}_{gamma}.jpg'
                cv2.imwrite(os.path.join(self.out_dir, fname), degraded_image)