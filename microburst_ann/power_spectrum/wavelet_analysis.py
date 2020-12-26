import datetime
import operator

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib
import math as m
from mpl_toolkits.axes_grid1 import make_axes_locatable

import wavelet_functions


class WaveletDetector():
    def __init__(self, data, time, cadence, **kwargs):
        """
        Initialize the wavelet parameters
        """
        self.dataCopy = data
        self.data = (data - np.mean(data)) / np.std(data, ddof=1)
        self.n = len(self.data)
        self.time = time
        self.cadence = cadence
        
        self.mother = kwargs.get('mother', 'DOG')
        self.wavelet_param = kwargs.get('wavelet_param', -1)
        self.j1 = kwargs.get('j1', 80)
        self.pad = kwargs.get('pad', 1)
        self.dj = kwargs.get('dj', 0.125)
        self.s0 = kwargs.get('s0', 2*cadence)
        self.siglvl = kwargs.get('siglvl', 0.98)
               
        self.lag1 = self.lagNAutoCorr(data, 1)
        
        # Run the microburst detection scipt with a wkarg keyword
        if kwargs.get('run_scipt', False):
            self.waveletTransform() # Transform data into wavelet space.
            self.waveletFilter(self.s0, 0.5) # Apply a high-pass and significance filter
            self.degenerateInvWaveletTransform() # Inverse tranform the leftovers to time-count space.
            self.TestForMicrobursts(COUNT_THRESH = 0.0) # Apply microburst test.
            print('Done detecting microbursts. Use the class indicies array.')
        return
        
    def waveletTransform(self):
        # Wavelet transform:
        self.wave, self.period, self.scale, self.coi = \
            wavelet_functions.wavelet(self.data, self.cadence, self.pad, 
                                    self.dj, self.s0, self.j1, self.mother,
                                    param=self.wavelet_param)
    
        if len(self.time) != len(self.coi):
            self.coi = self.coi[1:]
        
        self.power = (np.abs(self.wave)) ** 2  # compute wavelet power spectrum

        # Significance levels: (variance=1 for the normalized data)
        signif = wavelet_functions.wave_signif(([1.0]), dt=self.cadence, sigtest=0, scale=self.scale, \
            lag1=self.lag1, mother=self.mother, siglvl = self.siglvl)
        self.sig95 = signif[:, np.newaxis].dot(np.ones(self.n)[np.newaxis, :])  # expand signif --> (J+1)x(N) array
        self.sig95 = self.power / self.sig95  # where ratio > 1, power is significant
        return
        
    def waveletFilter(self, lowerPeriod, upperPeriod):
        """
        NAME:    waveletFilter(wave, sig95, l_s = 0, u_s = 29)
        USE:     This does a band pass and power filtering on the wavelet period-frequency domain. Give it the                   wavelet amplitude array, the power significance array, and the upper and lower scales that will                         be used for filtering. 
        RETURNS: The filtered wavelet amplitude array.
        MOD:     2016-04-13
        
        # WAVELET SCALE CALCULATOR ################# 
        # jth scale indicie = ln(S/S0)/(dJ * ln(2))
        ##################################    
        """
        lowerScale = int(np.floor(m.log(lowerPeriod/self.s0)/(self.dj*m.log(2))))
        upperScale = int(np.floor(m.log(upperPeriod/self.s0)/(self.dj*m.log(2))))
        
        self.waveFlt = self.wave
        
        # Band pass filter
        # Zero out parts of the wavlet space that we don't want to reconstruct. 
        self.waveFlt[upperScale:, :] = 0
        self.waveFlt[:lowerScale, :] = 0
    
        # Significance filter. Only pass data that has was significant above 
        # the red noise level defined in self.siglvl
        notSigInd = np.where(self.sig95 < 1) 
        self.waveFlt[notSigInd] = 0
        return self.waveFlt
        
    def degenerateInvWaveletTransform(self, waveFlt = None, C_d = 3.541, psi0 = 0.867):
        """
        Supply own C_d and psi0 if not using a DOG m = 2 wavelet.
        """
        if waveFlt == None:
            waveFlt = self.waveFlt
            
        InvTranform = np.zeros(self.n)
        tansformConstant = ((self.dj*m.sqrt(self.cadence))/(C_d*psi0) ) # Reconstruction constant. 
        
        # For more information, see article: "A Practical Guide to Wavelet Analysis", C. Torrence and G. P. Compo, 1998.
        for i in range(waveFlt.shape[0]):
            waveFlt[i, :] /= m.sqrt(self.period[i])
        for i in range(self.n):
            InvTranform[i] = np.sum(np.real(waveFlt[:, i]), axis = 0)
        self.dataFlt = tansformConstant*InvTranform
        return self.dataFlt
        
    def TestForMicrobursts(self, dataFlt = None, **kwargs):
        """
        Parameters:
        COUNT_THRESH = 0.05 # Absolute minimum count threshold of filtered data for microburst detection.
        CONCAVITY_THRESH = 0.3 # Max allowable fractional difference to be classified as a microburst. 
        TIME_THRESH = 1 # In seconds.
        """
        if dataFlt == None:
            dataFlt = self.dataFlt
            
        self.indicies = np. array([], dtype = int)
        tDiff = np.zeros(len(self.time)-1)
        
        COUNT_THRESH = kwargs.get('COUNT_THRESH', 0.05)
        TIME_THRESH = kwargs.get('TIME_THRESH', 1.00)
        
        DATA_GAP_THRESH = int(TIME_THRESH/self.cadence)
        
        if (isinstance(self.time[0], datetime.datetime) or isinstance(self.time[0], pd.DatetimeIndex)):
            for i in range(len(self.time)-1):
                tDiff[i] = (self.time[i+1] - self.time[i]).total_seconds()
        else:
            tDiff = np.abs(np.convolve([-1, 1], self.time, mode = 'same'))
    
        # Now determine which events are microbursts and which ones are false positives.
        #dropoutFlag = flag_dropouts.dropOutFlag(self.dataCopy)
        for i in range(DATA_GAP_THRESH, int(len(self.dataCopy) - DATA_GAP_THRESH)):
                # Apply the microburst filters. 
                # The 2*cadence is there since the change in data time stamps may
                # normally be around 30 ms for 18.75 ms cadence.
                if (
                    dataFlt[i] > COUNT_THRESH \
                    and np.abs(max(tDiff[(i - DATA_GAP_THRESH):(i + DATA_GAP_THRESH)])) < 2*self.cadence \
                    and self.dataCopy[i] > 100
                    ):
                    # these are the good detections. 
                    self.indicies = np.append(self.indicies, i)
        return self.indicies
        
        
    def findMicroburstPeaks(self, indicies = np.array([])):
        """
        NAME:    findMicroburstPeaks(self, indicies = np.array([]))
        USE:     Calculates the peak of each microburst from an array containing 
                 continious indicies that matched the microburst detection criterion. 
        RETURNS: An index array of detected microburst peaks.
        MOD:     2016-06-16
        """
        if len(indicies) == 0:
            indicies = self.indicies
            
        self.peaks = np.array([], dtype = int)
        self.startInd = np.array([], dtype = int)
        self.endInd = np.array([], dtype = int)
        endValue = 0
        startValue = 0
        indicies = np.append(indicies, 0) # This added number is from my imperfect algorithm.
        
        while(startValue < len(indicies)-1):
            while(indicies[endValue+1] - indicies[endValue] == 1):
                endValue += 1
            else:
                index, value = \
                max(enumerate(self.data[indicies[startValue]:indicies[endValue]+1]), key=operator.itemgetter(1))
                self.peaks = np.append(self.peaks, indicies[index+startValue])
                self.startInd = np.append(self.startInd, indicies[startValue])
                self.endInd = np.append(self.endInd, indicies[endValue]+1)
                endValue += 1
                startValue = endValue
        return self.peaks
        
    def plotPower(self, ax=None):
        """
        ax is the subplot argument.
        """
        self.levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
        
        if ax == None:
            f = plt.figure()    
            f, ax = plt.subplots(1)
        else:
            ax = np.ravel(ax)[0]
            
        # Max period is fourier_factor*S0*2^(j1*dj), fourier_factor = 3.97383530632 
        CS = ax.contourf(self.time, self.period, np.log2(self.power), len(self.levels))
        ax.fill_between(self.time, np.max(self.period), self.coi, alpha=0.8, 
            facecolor='white', zorder=3)
        im = ax.contourf(CS, levels=np.log2(self.levels))
        ax.set_xlabel('UTC')
        ax.set_ylabel('Period [s]')
        # 95# significance contour, levels at -99 (fake) and 1 (95# signif)
        ax.contour(self.time, self.period, self.sig95, [-99, 1], colors='k')
        # cone-of-influence, anything "below" is dubious
        ax.plot(self.time, self.coi, 'k')
        # format y-scale
        ax.set_yscale('log', basey=2, subsy=None)
        ax.set_ylim([np.min(self.period), np.max(self.period)])
        axy = ax.yaxis # plt.gca().yaxis
        axy.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.ticklabel_format(axis='y', style='plain')
        ax.invert_yaxis()
        # set up the size and location of the colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.5)
        plt.colorbar(im, cax=cax, orientation='horizontal')
        #######################################

    def lagNAutoCorr(self, x, n):
        """
        NAME:    lagNAutoCorr(x, n)
        USE:     Call it with an integer n, to get a lag-n autocorrelation 
                 for the array of data points x. 
        RETURNS: Returns the lag-n autocorrelation of list x. length of output list is n less than the input list.
        MOD:     2016-04-13
        """
        y = np.array(x[n:])
        z = np.array(x[:(len(x)-n)])
        result = np.mean( (z - np.mean(z)) * (y - np.mean(y)))/(np.std(y)*np.std(z))
        return result
        
    def savePeakIndicies(self, fdir, fname):
        np.save(fdir + fname, self.peaks, allow_pickle = False)
        return