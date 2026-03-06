
# Python standard library
import inspect
import os, sys
import importlib
import time
from importlib import reload
from tqdm import tqdm

# Other libraries
import scipy
import scipy.optimize
import numpy as np

from pylab import *


    
def Uniform(x,A):
    if type(x) not in [np.ndarray,list]:
        return A
    else:
        return A*np.ones_like(x)
def Exp(x,A,t):
    return A*np.exp(-x/t)
def Gauss(x, A, mean, sigma):
    return A * np.exp(-(x - mean)**2 / (2 * sigma**2)) 
def Gauss_sideband(x, A, mean, sigma, a1,a2):
    # a1 for left, a2 for right
    return Utils.Gauss(x, A, mean, sigma) + sqrt(2*np.pi)*sigma/2*(a1*scipy.special.erfc((x-mean)/sqrt(2)/sigma) + a2*(2-scipy.special.erfc((x-mean)/sqrt(2)/sigma))) 
def Poisson(k, Lambda,A):
    # Lambda: mean, A: amplitude
    return A*(Lambda**k/scipy.special.factorial(k)) * np.exp(-Lambda)
def Poly(x, *P):
    '''
    Compute polynomial P(x) where P is a vector of coefficients
    Lowest order coefficient at P[0].  Uses Horner's Method.
    '''
    result = 0
    for coeff in P[::-1]:
        result = x * result + coeff
    return result

def Pulse(x, A, x0 = 0, tau1=2, tau2=20):
    # Fs: samples per s
    dx=(x-x0)
    dx*=np.heaviside(dx,1)
    kernel = (np.exp(-dx/tau1)-np.exp(-dx/tau2))/(tau1-tau2)*np.heaviside(dx,1)
    kernel_normed = kernel/np.max(kernel)
    return kernel_normed*A

def Pulse2(x, tau_r, tau_f1, tau_f2, A1, A2, t0, A0):
    # Test: plot(np.linspace(-50,50,200), Pulse3(np.linspace(-50,50,200), 2,4,10,20,3,2,1,10.5,5))
    times=x-t0
    mask =times>0
    pulse = (A1     *     (np.exp(-times[mask] / tau_f1))
            +A2     *     (np.exp(-times[mask] / tau_f2))
            -(A1+A2) * (np.exp(-times[mask] / tau_r))
            )
    pulse = np.concatenate((np.zeros(sum(~mask)), pulse))
    pulse/=max(pulse)
    pulse*=A0
    return pulse

def Pulse3(x, tau_r, tau_f1, tau_f2, tau_f3, A1, A2, A3, t0, A0):
    # Test: plot(np.linspace(-50,50,200), Pulse3(np.linspace(-50,50,200), 2,4,10,20,3,2,1,10.5,5))
    times=x-t0
    mask =times>0
    pulse = (A1     *     (np.exp(-times[mask] / tau_f1))
            +A2     *     (np.exp(-times[mask] / tau_f2))
            +A3     *     (np.exp(-times[mask] / tau_f3))
            -(A1+A2+A3) * (np.exp(-times[mask] / tau_r))
            )
    pulse = np.concatenate((np.zeros(sum(~mask)), pulse))
    pulse/=max(pulse)
    pulse*=A0
    return pulse

def Chi2(x, dof, A):
    return scipy.stats.chi2.pdf(x,dof)*A 


def pulse_2pole(t_rise_ns, t_fall_ns, samples_per_ns, total_samples=8000, pre_trig_samples=0):
    total_time = total_samples/samples_per_ns
    x = np.linspace(0,total_time, total_samples)
    x0 = pre_trig_samples/samples_per_ns
    dx=(x-x0)
    dx*=np.heaviside(dx,1)
    kernel = (np.exp(-dx/t_fall_ns)-np.exp(-dx/t_rise_ns))/np.heaviside(dx,1)
    kernel_normed = kernel/max(kernel)#(np.dot(kernel,kernel/max(kernel)))
    
    return kernel_normed


def pulse_2pole_old(pre_trig, post_trig, tau1=2,tau2=20, Fs = 100):
    # Fs: samples per s
    x = np.arange(0,(pre_trig+post_trig))/Fs
    x0 = pre_trig/Fs
    dx=(x-x0)
    dx*=np.heaviside(dx,1)
    kernel = (np.exp(-dx/tau1)-np.exp(-dx/tau2))/(tau1-tau2)*np.heaviside(dx,1)
    # kernel_normed = kernel/(np.dot(kernel,kernel/max(kernel)))
    kernel_normed = kernel/np.max(kernel)
    return kernel_normed

def roll_zeropad(a, shift):
    y = np.roll(a,shift)
    y[:shift]=0
    return y

def slope(y, x=None):
    x = np.arange(len(y)) if x is None else x
    slope,*_ = scipy.stats.linregress(x, y)
    return slope









def fstr(template, scope):
    """
    Evaluate the f string later at a given scope
    """
    return eval(f"f'{template}'", scope)  

def fit_curve(f, xdata, ydata, makeplot = True, label=None, fit_range=None, p0=None, sigma=None, absolute_sigma=False, check_finite=None, bounds=(-np.inf, np.inf), method=None, jac=None, **kwargs):
   
    if type(f) is str:
        if f in ["Gauss","gauss","gaus"]:
            f=Gauss
            mean = np.sum(xdata*ydata)/np.sum(ydata)
            p0 = [np.max(ydata), mean, np.sqrt(np.sum(ydata*(xdata-mean)**2)/(np.sum(ydata)-1))] if p0 is None else p0
        elif f in ["Exp", "exp"]:
            f=Exp
            
    
    # Fit range
    fit_range = [np.min(xdata), np.max(xdata)] if fit_range is None else fit_range
    mask = (xdata>=fit_range[0]) &(xdata<=fit_range[1])
    xdata = xdata[mask]
    ydata = ydata[mask]   
    
    # Get keyword arguments:
    curvefit_kwargs = {}
    for key, value in kwargs.items() :
        if key in inspect.getfullargspec(scipy.optimize.curve_fit)[0] or key in ["maxfev"] :
            curvefit_kwargs[key] = value
    # for key, value in kwargs.items():    
            
    popt, pcov, info, *_ = scipy.optimize.curve_fit(f, xdata, ydata, p0=p0, sigma=sigma, absolute_sigma=absolute_sigma, check_finite=check_finite, bounds=bounds, method=method, jac=jac, full_output=True, **curvefit_kwargs)   
    
    if makeplot:
        xdata_plot = np.linspace(*fit_range, 200)
        ydata_plot = f(xdata_plot, *popt)
        
        scope = locals()
        func_parameter_names = inspect.getfullargspec(f)[0][1:]
        label = fstr(label,scope) if label is not None else "Fit"
        
        # Get keyword arguments:
        plotkwargs = {}
        for key, value in kwargs.items() :
            if key in ["color", "marker", "linestyle"] :
                plotkwargs[key] = value
        # for key, value in kwargs.items():
        plot(xdata_plot, ydata_plot, label=label, **plotkwargs)    
    
    return popt, pcov, info, f
    
    
def fit_hist(f, h, makeplot = True, label=None, fit_range=None, p0=None, sigma=None, absolute_sigma=False, check_finite=None, bounds=(-np.inf, np.inf), method=None, jac=None, full_output=False, nan_policy=None, **kwargs):

    
    # Limit fit range
    xdata = 0.5*(h[1][:-1]+h[1][1:])
    ydata = h[0]
    fit_range = [xdata[0], xdata[-1]] if fit_range is None else fit_range
    mask = (xdata>=fit_range[0]) &(xdata<=fit_range[1])
    xdata = xdata[mask]
    ydata = ydata[mask]
    
    # Calculate uncertainty
    sigma = np.sqrt(ydata) if sigma is None else sigma; sigma[sigma==0] =1
    
    # Fit
    popt, pcov, info, f = fit_curve(f, xdata, ydata, makeplot = makeplot, label=label, p0=p0, sigma=sigma, absolute_sigma=absolute_sigma, check_finite=check_finite, bounds=bounds, method=method, jac=jac, **kwargs)

    return popt, pcov, info, f





from typing import List, Tuple
import scipy.ndimage

def constant_fraction_discriminator(waveform: List[float], baseline: float, threshold: float, fraction: float, gauss_filter = None) -> List[Tuple[float, int]]:
    """
    This function takes in a waveform, baseline, threshold, and fraction as input and returns a list of tuples containing
    the amplitude and sample number of the leading edge of each pulse.
    """
    leading_edges = []
    triggered = False
    
    if gauss_filter is not None:
        waveform = scipy.ndimage.gaussian_filter(waveform,sigma=gauss_filter,)
    
    for i in range(1, len(waveform)):
        if waveform[i] > baseline + threshold:
            if triggered:
                continue
                
            for j in range(i - 1, -1, -1):
                if waveform[j] < baseline + fraction * (waveform[i] - baseline):
                    leading_edges.append((waveform[i], i))
                    triggered=True
                    break
        else:
            triggered = False
    return leading_edges



# trace = data_save[1][0]
# trace-=np.mean(trace[:1600])
# trace = -trace
# trace/=15.8
# leading_edges = constant_fraction_discriminator(trace, 0, 0.02, 0.5, gauss_filter=4)
# plot(trace)
# print(leading_edges)
# axvline(leading_edges[0][1])


    
def float_to_ADU(waveform, bits=14):
    # SCPIcmd = ":DATA1 VOLATILE, "
    # strData = ""
    # for i in waveform:
    #     strData+=f"{i:.7f},"
    # strData=strData[:-1]   
    # SCPIcmd = SCPIcmd + strData
    waveform_new=waveform/np.max(np.abs(waveform))
    waveform_new=np.round(waveform_new* 2**(bits-1))
    return waveform_new    





# Generate white noise
def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real

def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    """
    Generate band limited noise
    The returned noise density is 1/rtHz
    """
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
    f[idx] = 1
    return fftnoise(f)*np.sqrt(samples*samplerate/2)



class Cut:
    # def __init__(self):
        # self.utils=Utils()

    class _UnbiasedEstimators(object):
        """
        //From QETpy: https://github.com/ucbpylegroup/QETpy
        Helper class for calculating the unbiased estimators of a 1D normal
        distribution that has been truncated at specified bounds.
        Attributes
        ----------
        mu0 : float
            The biased estimator of the mean of the inputted and truncated data.
        std0 : float
            The biased estimator of the standard deviation of the inputted and
            truncated data.
        mu : float
            The unbiased estimator of the mean of the inputted and truncated data.
        std : float
            The unbiased estimator of the standard deviation of the inputted and
            truncated data.
        """

        def __init__(self, x, lwrbnd, uprbnd):
            """
            Initialization of the `_UnbiasedEstimators` helper class
            Parameters
            ----------
            x : ndarray
                A 1D array of data that has been truncated, for which the unbiased
                estimators will be calculated.
            lwrbnd : float
                The lower bound of the truncation of the distribution.
            uprbnd : float
                The upper bound of the truncation of the distribution.
            """

            inds = (np.asarray(x) >= lwrbnd) & (np.asarray(x) <=uprbnd)

            self._lwrbnd = lwrbnd
            self._uprbnd = uprbnd

            self._x = x[inds] # make sure data is only between the specified bounds
            self._sumx = np.sum(self._x)
            self._sumx2 = np.sum(self._x**2)
            self._lenx = len(self._x)

            self.mu0 = np.mean(self._x)
            self.std0 = np.std(self._x)

            self._calc_unbiased_estimators()

        def _equations(self, p):
            """
            Helper method for calculating the system of equations that will be
            numerically solved for find the unbiased estimators.
            Parameters
            ----------
            p : tuple
                A tuple of length 2 containing the current estimated values of the
                unbiased estimators: (mu, std).
            Returns
            -------
            (mu_eqn, std_eqn) : tuple
                A tuple containing the two equations that will be solved to give the
                unbiased estimators of the mean and standard deviation of the data.
            """

            mu, std = p

            pdf_lwr = stats.norm.pdf(self._lwrbnd, loc=mu, scale=std)
            pdf_upr = stats.norm.pdf(self._uprbnd, loc=mu, scale=std)

            cdf_lwr = stats.norm.cdf(self._lwrbnd, loc=mu, scale=std)
            cdf_upr = stats.norm.cdf(self._uprbnd, loc=mu, scale=std)

            mu_eqn = self._sumx - self._lenx * mu
            # term due to truncation
            mu_eqn += self._lenx / (cdf_upr - cdf_lwr) * (pdf_upr - pdf_lwr)

            std_eqn = self._sumx2 - 2 * mu * self._sumx + self._lenx * mu**2 - self._lenx * std**2
            # term due to truncation
            std_eqn += self._lenx * std**2 / (cdf_upr - cdf_lwr) * ((self._uprbnd - mu) * pdf_upr - (self._lwrbnd - mu) * pdf_lwr)

            return (mu_eqn, std_eqn)

        def _calc_unbiased_estimators(self):
            """
            Method for calculating the unbiased estimators of the truncated distribution.
            """

            self.mu, self.std = scipy.optimize.fsolve(self._equations, (self.mu0, self.std0))

    def removeoutliers(self, data, skew_target=0.05, skew_itermax=20, std_target=3, std_itermax=20, std_precision=1000, return_unbiased_estimates=False, axis=0, verbose=False):
        """
        Removing outliers by
          1) skewness (minimizing to skewtarget)
          2) standard deviation

        Parameters
            x : array
                Array of real-valued variables from which to remove outliers.
            skew_target:
        Returns
            inds : ndarray
                Boolean indices
        """
        # 0) Adapt to data type
        if type(data) is list:
            data=np.array(data)
        if data.ndim==1:
            x=data


        # 1) Skewness cut
        if skew_target is not None:
            i=1
            inds=(x != np.inf)
            sk=scipy.stats.skew(x[inds])
            while(abs(sk) > skew_target):
                dmed=x-np.median(x[inds])
                dist=np.min([abs(min(dmed)),abs(max(dmed))])
                inds=inds & (abs(dmed) < dist)
                sk=scipy.stats.skew(x[inds])
                if(i > skew_itermax):
                    if verbose:
                        print(f"Skew Reaching maximum {skew_itermax} iterations. Stop at skew of {sk}")
                    break
                i+=1
        else:
            inds=np.ones(len(x),dtype=bool)

        # 2) Standard deviation cut
        if std_target is not None:
            # Trun the relative precision into absolute, in the unit of standard deviation
            std_precision_abs = np.std(x)/std_precision
            mean_last = np.mean(x[inds]); std_last = np.std(x[inds]);
            i=0
            nstable = 0

            while nstable <= 3:
                mask = inds& (abs(scipy.stats.zscore(x)) < std_target)
                if sum(mask) <=1 and verbose:
                    warnings.warn(
                        "The number of events passing iterative cut via iterstat is <= 1. "
                        "Iteration not converging properly. Returning simple mean and std. "
                        "No data will be cut."
                    )
                    mask = inds&np.ones(len(x),dtype=bool)
                    mean_this = np.mean(x[mask])
                    std_this = np.std(x[mask])
                    break

                mean_this = np.mean(x[mask])
                std_this = np.std(x[mask])

                if (abs(mean_this - mean_last) > std_precision_abs) or (abs(std_this - std_last) > std_precision_abs):
                    nstable = 0
                else:
                    nstable = nstable + 1

                mean_last = mean_this
                std_last = std_this

                i+=1
                if(i > std_itermax):
                    if verbose:
                        print(f"STD Reaching maximum {std_itermax} iterations. Stop at standard deviation of {std_last}")
                    break

        # 3. Calculate mean and stddev
        if return_unbiased_estimates:
            unb = self._UnbiasedEstimators(x[mask], mean_this - std_target * std_this, mean_this + std_target * std_this)
            mean_this=unb.mu; std_this=unb.std

        return mask, mean_this, std_this


class Trigger:
    @staticmethod
    def threshold_trigger(trace, trig_th, rising_edge=True,
                           align_max=False, deactivate_th=None, peak_search_window_limit=4096, trigger_holdoff=-1):
        """
        Run the threshold trigger, align to the maximum point after the trigger cross-over
        Return trigger points, trigger_amplitude and trigger width

        Inputs
        -------
        trace: array
            The trace to be triggered
        
        trig_th: float
            Trigger threshold
        
        rising_edge: boolean
            Trigger on rising edge if True, on falling edge if False.
        
        align_max: boolean
            Align the peaks if True
        
        deactivate_th: float
            Deactivate threshold: close the window for trigger point search. It's the multiple of trig_th.
            Example: if trig_th is 1.8eV and you want deactivate threshold to be 0.9eV, set deactivate_th=0.5
            Defualt setting None if you want deactivate_th=trig_th
        
        peak_search_window_limit:int
            The limit for peak search window
        
        trigger_holdoff: int
            The counts (*not time*) before next trigger point

        Returns
        -------
        trigger_points,trigger_amplitude,trigger_widths
        """
        if rising_edge:
            trigger_points=np.flatnonzero((trace[0:-1] < trig_th) & (trace[1:] > trig_th))+1
        else:
            trigger_points=np.flatnonzero((trace[0:-1] > trig_th) & (trace[1:] < trig_th))+1


        if align_max is False:
            trigger_points_final=trigger_points
            trigger_amp_final=np.zeros(len(trigger_points))
            trigger_widths_final=np.zeros(len(trigger_points))
        # Align to the maximum point after the trigger cross-over
        else:
            if deactivate_th==None:
                deactivate_th=trig_th
            else:
                deactivate_th*=trig_th
            trigger_points_aligned = []
            trigger_amp_aligned = []
            trigger_widths = []
            for trigpt in trigger_points:
                if trigpt<(len(trace)-peak_search_window_limit):
                    # Set the peak search window, from the trigger point to where the trace goes below deactivate_th
                    search_window = np.argmax(trace[trigpt:trigpt+peak_search_window_limit]<deactivate_th)
                    if search_window==0:
                        search_window=peak_search_window_limit
                    # Adjust trigger point to maximum
                    offset = np.argmax(trace[trigpt:trigpt+search_window])
                    trigger_points_aligned.append(trigpt+offset)
                    trigger_amp_aligned.append(trace[trigger_points_aligned[-1]])
                    trigger_widths.append(search_window)
            # Remove possible duplications:
            trigger_points_aligned,mask = np.unique(trigger_points_aligned,return_index=True)
            trigger_points_final = trigger_points_aligned.astype(int)
            trigger_amp_final = np.array(trigger_amp_aligned)[mask]
            trigger_widths_final = np.array(trigger_widths).astype(int)[mask]

        if trigger_holdoff>0:
            trigger_holdoff=int(trigger_holdoff)
            mask_holdoff = np.abs(Utils.diff_coinc(trigger_points_final,use_nearest=False)[0])>trigger_holdoff # Reject events with dt<trigger_holdoff
            trigger_points_final=trigger_points_final[mask_holdoff]
            trigger_amp_final=trigger_amp_final[mask_holdoff]
            trigger_widths_final=trigger_widths_final[mask_holdoff]

        return trigger_points_final, trigger_amp_final, trigger_widths_final
    


def get_baseline_info(traces):
    """
    Calculate mean, max, std, slope of all traces
    """
    time_series = np.arange(len(traces[0]))
    def get_trace_info(trace):
        trace_means = np.mean(trace)     
        trace_max = np.max(trace)   
        trace_std = np.std(trace)     
        trace_slope,*_= scipy.stats.linregress(time_series, trace)  
        trace_skew = scipy.stats.skew(trace)
        return [trace_means,trace_max,trace_std,trace_slope,trace_skew ]
    trace_info = np.apply_along_axis(get_trace_info, 1, traces)
    return trace_info   


def get_pulses_info(traces, pre_trig=None, threshold_in_sigma=6, gaussian_filter_sigma=2):
    """
    Calculate information on baseline part and pulse part:
    * mean, max, std, slope of the pre-trigger region
    * the number of triggers, and the trigger point on pulse

    INPUT:
        pre_trig: int, index of the pre trigge region
        gaussian_filter_sigma: int or None. gaussian_filter sigma. Set to None to disable.

    """
    
    time_series = np.arange(len(traces[0]))
    pre_trig = int(0.48*len(traces[0])) if pre_trig is None else int(pre_trig)

    # Set the trigger threshold
    baseline_concatenated = np.concatenate((traces[:min(20,len(traces))]))
    if gaussian_filter_sigma is not None:
        baseline_concatenated = scipy.ndimage.gaussian_filter(baseline_concatenated, gaussian_filter_sigma, order=1)
    trigger_threshold = np.std(baseline_concatenated)*threshold_in_sigma

    
    def get_trace_info(trace):
        trace_means = np.mean(trace[:pre_trig])     
        trace_max = np.max(trace[:pre_trig])   
        trace_std = np.std(trace[:pre_trig])     
        trace_slope,*_= scipy.stats.linregress(time_series[:pre_trig], trace[:pre_trig])  
        trace_skew = scipy.stats.skew(trace[:pre_trig])

        trace_for_trigger = trace if gaussian_filter_sigma is None else scipy.ndimage.gaussian_filter(trace, gaussian_filter_sigma, order=1)
        trigger_points_final, trigger_amp_final, trigger_widths_final = Trigger.threshold_trigger(trace_for_trigger, trigger_threshold)
        trace_pileups = len(trigger_points_final)
        trace_trigger_point = trigger_points_final[0] if len(trigger_points_final)>0 else -1
        trace_amp = max(trace[trace_trigger_point:]) - trace_means
        return [trace_means,trace_max,trace_std,trace_slope,trace_skew, trace_pileups, trace_trigger_point, trace_amp]
    
    
    trace_info = np.apply_along_axis(get_trace_info, 1, traces)
    return trace_info   

def cut_baseline(traces, cut_iterations = 2):
    cut = Cut()
    trace_info = get_baseline_info(traces)

    inds_keep = np.arange(len(traces))
    cut_passage_fraction = []
    nevents = len(traces)
    for iter in range(cut_iterations):
        mask_all = np.ones(len(inds_keep)).astype(bool)

        # Do all other cuts for baseline
        mask_mean   = cut.removeoutliers(trace_info[:,0])[0]
        mask_max   = cut.removeoutliers(trace_info[:,1])[0]
        mask_std   = cut.removeoutliers(trace_info[:,2])[0]
        mask_slope = cut.removeoutliers(trace_info[:,3])[0]
        mask_skew  = cut.removeoutliers(trace_info[:,4])[0]


        inds_keep = inds_keep[mask_all]
        trace_info = trace_info[mask_all]
        mask_all = mask_mean & mask_max & mask_std & mask_slope & mask_skew
        cut_passage_fraction.append({"mean":     sum(mask_mean)/nevents,
                                     "max":  sum(mask_max)/nevents,
                                     "std":  sum(mask_std)/nevents,
                                     "slope":    sum(mask_slope)/nevents,
                                     "skew":     sum(mask_skew)/nevents,
                                     "combined":     sum(mask_all)/nevents,})

    return inds_keep, cut_passage_fraction


def cut_pulses(traces, cut_amp = None, cut_baseline="auto", cut_triggerpoint = True, cut_pileups= True, cut_iterations = 2, pre_trig=None, threshold_in_sigma=6, gaussian_filter_sigma=2):
    """
    Do auto and manual cut on pulses.

    If use manual cut, specify the amplitude and baseline range
    """
    cut = Cut()
    trace_info = get_pulses_info(traces, pre_trig=pre_trig, threshold_in_sigma=threshold_in_sigma, gaussian_filter_sigma=gaussian_filter_sigma)

    inds_keep = np.arange(len(traces))
    cut_passage_fraction=[]
    nevents = len(traces)
    for iter in range(cut_iterations):
        mask_all = np.ones(len(inds_keep)).astype(bool)

        # Do all other cuts for baseline
        mask_max   = cut.removeoutliers(trace_info[:,1])[0]
        mask_std   = cut.removeoutliers(trace_info[:,2])[0]
        mask_slope = cut.removeoutliers(trace_info[:,3])[0]
        mask_skew  = cut.removeoutliers(trace_info[:,4])[0]
        mask_pileup = mask_triggerpoint = mask_amp = mask_all

        # if enabled, do pileup cut only once
        if iter==0 and cut_pileups:
            mask_pileup = (trace_info[:,5]<= int(cut_pileups))

        # if enabled, do trigger point cut only once
        if iter==0 and cut_triggerpoint:
            mask_triggerpoint = cut.removeoutliers(trace_info[:,6])[0]            

        # if enabled, do auto baseline cut
        if cut_baseline=="auto":
            mask_baseline = cut.removeoutliers(trace_info[:,0])[0]
        else:
            mask_baseline = (trace_info[:,0]>cut_baseline[0]) & (trace_info[:,0]<cut_baseline[1])

        # if enabled, do amplitude cut
        if cut_amp is not None:
            mask_amp = (trace_info[:,7]>cut_amp[0]) & (trace_info[:,7]<cut_amp[1])
        
        
        mask_all = mask_baseline & mask_max & mask_std & mask_slope & mask_skew & mask_pileup & mask_triggerpoint & mask_amp
        inds_keep = inds_keep[mask_all]
        trace_info = trace_info[mask_all]
        cut_passage_fraction.append({"baseline_mean":     sum(mask_baseline)/nevents,
                                     "baseline_max":  sum(mask_max)/nevents,
                                     "baseline_std":  sum(mask_std)/nevents,
                                     "baseline_slope":    sum(mask_slope)/nevents,
                                     "baseline_skew":     sum(mask_skew)/nevents,
                                     "pileups":     sum(mask_pileup)/nevents,
                                     "trigger_point":     sum(mask_triggerpoint)/nevents,
                                     "amplitude":     sum(mask_amp)/nevents,
                                     "combined":     sum(mask_all)/nevents,})        

    return inds_keep,cut_passage_fraction


def make_avg_pulse(traces, cut_amp = None, cut_baseline="auto", cut_triggerpoint = True, cut_pileups= True, cut_iterations = 2, pre_trig=None, threshold_in_sigma=6, gaussian_filter_sigma=2):
    inds_keep,cut_passage_fraction = cut_pulses(traces, cut_amp = cut_amp, cut_baseline=cut_baseline, cut_triggerpoint = cut_triggerpoint, cut_pileups= cut_pileups, cut_iterations = cut_iterations, pre_trig=pre_trig, threshold_in_sigma=threshold_in_sigma, gaussian_filter_sigma=gaussian_filter_sigma)
    return np.mean(traces[inds_keep], axis=0)
