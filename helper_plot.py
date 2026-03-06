import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl



def plotdark(plot_fn, fig, background_col=(0.1, 0.1, 0.1), face_col=(0, 0, 0)):
    """
    Apply a dark theme to a matplotlib figure.

    Parameters:
    plot_fn (function): A lambda function containing plot commands.
    fig (Figure): The matplotlib figure object.
    background_col (tuple): RGB values for the background color.
    face_col (tuple): RGB values for the face color of the plot.

    Returns:
    None
    """
    fig.patch.set_facecolor(background_col)
    plot_fn()
    ax = plt.gca()
    ax.set_facecolor(face_col)
    ax.spines['bottom'].set_color('k')
    ax.spines['top'].set_color('k')
    ax.spines['left'].set_color('k')
    ax.spines['right'].set_color('k')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.grid(alpha=0.1)
    ax.title.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

class plot_phos:
    """
    A class for generating phosphor-style plots with glow effects.

    ```example
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Generate waveform data
    x = np.linspace(0, 1, 100000)
    y = np.sin(10000 * x) * (np.sin(20*3 * x+2) + 2)
    y2 = np.sin(10000 * x) * (np.sin(7*3 * x+2) + 2)
    
    f = helper_plot.plot_phos(sigma=1, upsample=10, bins=(600,400), figsize=(6,4), grid_alpha=0.1, dark_theme=True)
    f.add(x,y)
    f.add(x,y2*0.5-3)
    f.show()

    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")    
    plt.ylim(-5,5)
    ```
    """
    def __init__(self, sigma=1, upsample=10, bins=(600, 400),
                 figsize=(6, 4), grid_alpha=0.1, dark_theme=True, alpha=1):
        """
        Initialize the plot_phos class with rendering parameters.
        
        Parameters:
        sigma (float): Gaussian blur intensity.
        upsample (int): Factor by which to upsample the signal.
        bins (tuple): Resolution of the 2D histogram (x, y).
        figsize (tuple): Figure size for matplotlib.
        grid_alpha (float): Transparency level of the grid.
        dark_theme (bool): Whether to apply a dark theme.
        alpha (float): Alpha blending value.
        """
        # Store universal parameters
        self.sigma = sigma
        self.upsample = upsample
        self.bins = bins
        self.figsize = figsize
        self.grid_alpha = grid_alpha
        self.dark_theme = dark_theme
        self.alpha = alpha
        
        # Internal data storage
        self.counter = 0
        self.data_x = {}
        self.data_y = {}
        self.data_lrange = {}
        self.extent = [np.inf, -np.inf, np.inf, -np.inf]
        self.images = []


    @staticmethod
    def plot(x, y, fig=None, upsample=10, sigma=1.0, alpha=1, lmin=0.05, lmax=1, bins=(600, 400),
                      extent=None, figsize=None, dpi=150, grid_alpha=0.1, dark_theme=True, cmap=0, show=True, norm="linear", brightness = 3):
        """
        Generate a phosphor-style plot from input data.
        
        Parameters:
        x (ndarray): X-axis data.
        y (ndarray): Y-axis data.
        fig (Figure): Optional figure object.
        upsample (int): Factor for signal upsampling.
        sigma (float): Gaussian blur intensity.
        alpha (float): Alpha transparency.
        lmin, lmax (float): Lower and upper intensity limits.
        bins (tuple): Resolution of the 2D histogram.
        extent (tuple): Plot limits (optional).
        figsize (tuple): Size of the figure.
        grid_alpha (float): Transparency of the grid.
        dark_theme (bool): Whether to apply a dark theme.
        cmap (int or colormap): Colormap selection. 0-3 corresponds to four predefined colors. 
            0 mimics the traditional look of CRT.
        show (bool): Whether to display the plot.

        Returns:
        Figure, Axes: If show=True, returns matplotlib figure and axes.
        ndarray, extent: If show=False, returns the image array and extent.
        """
        ## Arguments
        figsize = figsize if figsize is not None else [bins[0]/dpi, bins[1]/dpi]
        
        # Upsample using sinc interpolation
        x_highres = np.linspace(x.min(), x.max(), len(x) * upsample)
        y_highres = scipy.signal.resample(y, len(x) * upsample)
        
        # Define phosphor intensity grid
        xspan = x.max()-x.min()
        yspan = y.max()-y.min()
        extent = (x.min()-xspan*0.02, x.max()+xspan*0.02, y.min()-yspan*0.02, y.max()+yspan*0.02) if extent is None else extent
        heatmap, xedges, yedges = np.histogram2d(x_highres, y_highres, bins=bins, \
                                                 range=((extent[0], extent[1]),(extent[2],extent[3])))
    
        # Apply efficient glow effect
        image = scipy.ndimage.gaussian_filter(heatmap, sigma=sigma)
    
        # Limit the intensity range
        mask_bkg = (image==0)
        data_tmp = (image[~mask_bkg] / (image.max()*lmax) * (1-lmin) + lmin)
        # Normalize to [0,1]
        data_tmp[data_tmp>1]=1
        if norm=="linear":
            image[~mask_bkg] = data_tmp
        else:
            image[~mask_bkg] = np.log10(data_tmp)
            imin,imax = min(image[~mask_bkg]), max(image[~mask_bkg])
            image[~mask_bkg] = (image[~mask_bkg] - imin)/(imax-imin)

        image[~mask_bkg] = image[~mask_bkg]**(1/brightness)
    
        # Give the image color
        image_final = np.zeros((bins[0], bins[1], 3), dtype=np.uint8)
        if cmap==0:
            image = (image * 255).astype(np.uint8)
            # Create turquoise gradient mapping
            gradient = np.zeros((256, 3), dtype=np.uint8)
            for i in range(128):
                gradient[i] = [i // 2, int(i * 1.5), int(i * 1.5)]  # Dark cyan
            for i in range(128, 256):
                gradient[i] = [64 + int((i - 128) * 1.5), 192 + (i - 128) // 2, 192 + (i - 128) // 2]  # Bright cyan
            # Convert grayscale phosphor image to RGB using gradient
            image_final[~mask_bkg] = gradient[image[~mask_bkg]]
    
        else:
            if cmap == 1:
                cmap = plt.cm.Greys_r
            elif cmap == 2:
                cmap = plt.cm.afmhot    
            elif cmap == 3:
                cmap = plt.cm.bone                
            else:
                cmap = cmap
            
            # Normalize and apply colormap
            image_final[~mask_bkg] = (cmap(image[~mask_bkg])[:,:3]*255).astype(np.uint8) # Color map and delete alpha channel
            
    
        if show:
            # Display the phosphor-style waveform
            fig = plt.figure(figsize=figsize) if fig is None else fig
            ax = plt.gca()
            ax.set_facecolor("k")
            ax.grid(alpha=grid_alpha)
        
            if dark_theme:
                plotdark(lambda: ax.imshow(np.transpose(image_final, (1,0,2)), aspect='auto', origin='lower', extent=extent), fig)
            else:
                ax.imshow(np.transpose(image_final, (1,0,2)), aspect='auto', origin='lower', extent=extent, alpha=alpha)
            return fig, ax
    
        else:
            return image_final, ~mask_bkg, extent     
    
    def add(self, x, y, lmin=0.05, lmax=1):
        """
        Add data to the phosphor plot.
        
        Parameters:
        x (ndarray): X-axis data.
        y (ndarray): Y-axis data.
        lmin, lmax (float): Lower and upper intensity limits.
        """
        self.data_x[self.counter] = x
        self.data_y[self.counter] = y
        self.data_lrange[self.counter] = [lmin, lmax]
        self.extent[0] = min(self.extent[0], min(x))
        self.extent[1] = max(self.extent[1], max(x))
        self.extent[2] = min(self.extent[2], min(y))
        self.extent[3] = max(self.extent[3], max(y))
        self.counter += 1
    
    def show(self, fig=None, color=None):
        """
        Display the accumulated phosphor-style plots.
        
        Parameters:
        fig (Figure): Optional figure object.
        color (Bool or list of cmap): color of the traces.
            True: loop the four internal colors
            [list of plt.cm.xxx]
        
        Returns:
        Figure, Axes: The generated figure and axes.
        """
        # Display the phosphor-style waveform
        fig = plt.figure(figsize=self.figsize) if fig is None else fig
        ax = plt.gca()
        ax.set_facecolor("k")
        ax.grid(alpha=self.grid_alpha)    

        if type(color) is list:
            cmaps = color 
        elif  color is True:
            cmaps =  [0,1,2,3]
        else:
            cmaps = [0]
        
        for i in self.data_x:
            img, mask, extent = self.plot(self.data_x[i], self.data_y[i], upsample=self.upsample, sigma=self.sigma, alpha=1, \
                                    lmin = self.data_lrange[i][0], lmax = self.data_lrange[i][1], \
                                    bins=self.bins, extent=self.extent, show=False, cmap= cmaps[i%len(cmaps)])
            self.images.append(img)

        # Combine all images with "Screen" blending mode
        image_combined = np.zeros_like(self.images[0], dtype=float)
        for i in self.data_x:
            # self.image+=self.images[i]
            img2 = np.array(self.images[i], dtype=float) / 255
            image_combined = 1 - (1-image_combined) * (1-img2)


        image_combined = (image_combined*255).astype(int)
        
        if self.dark_theme:
            plotdark(lambda: ax.imshow(np.transpose(image_combined, (1,0,2)), aspect='auto', origin='lower', extent=extent, alpha=self.alpha), fig)
        else:
            ax.imshow(np.transpose(image_combined, (1,0,2)), aspect='auto', origin='lower', extent=extent, alpha=self.alpha)        

        return fig,ax
