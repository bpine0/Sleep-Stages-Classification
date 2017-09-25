import numpy as np
import math

class ButterworthFilter:
    def __init__(self, cutoffFrequency):
        self.sampleRate = 30

        self.ax = np.zeros(3)
        self.by = np.zeros(3)

        self.xv = None
        self.yv = None

        self.cutoffFrequency = cutoffFrequency

        self.xv = np.zeros((3,3))
        self.yv = np.zeros((3,3))

        self._getLPCoefficientsButterworth2Pole(self.sampleRate, self.cutoffFrequency)

    # Filter using butterworth filter
    def getFilteredValues(self, sample):
        result = np.zeros(3)

        for i in range(3):
            self.xv[i,2] = self.xv[i,1]
            self.xv[i,1] = self.xv[i,0]
            self.xv[i,0] = sample[i]
            self.yv[i,2] = self.yv[i,1]
            self.yv[i,1] = self.yv[i,0]

            self.yv[i,0] =   (self.ax[0] * self.xv[i,0] + self.ax[1] * self.xv[i,1] 
                + self.ax[2] * self.xv[i,2]
                    - self.by[1] * self.yv[i,0]
                            - self.by[2] * self.yv[i,1])

            result[i] = self.yv[i,0]

        return result

    # Get Butterworth 2 Pole LPC Coefficients
    def _getLPCoefficientsButterworth2Pole(self, sampleRate, cutoff):
        PI = 3.1415926535897932385
        sqrt2 = 1.4142135623730950488

        QcRaw  = (2 * PI * cutoff) / sampleRate # Find cutoff frequency in [0..PI]
        QcWarp = math.tan(QcRaw) # Warp cutoff frequency

        gain = 1 / (1+sqrt2/QcWarp + 2/(QcWarp*QcWarp))
        self.by[2] = (1 - sqrt2/QcWarp + 2/(QcWarp*QcWarp)) * gain
        self.by[1] = (2 - 2 * 2/(QcWarp*QcWarp)) * gain
        self.by[0] = 1
        self.ax[0] = 1 * gain
        self.ax[1] = 2 * gain
        self.ax[2] = 1 * gain

class ExponentialFilter:
    # Use this constructor to use an exponential (smoothing) filter
    def __init__(self, smoothFactor):
        self.smoothFactor = max([smoothFactor,1])
        self.expectedValue = [None, None, None]
    
    # Filter using Smoothing Filter
    def getFilteredValues(self, sample):
        for i in range(3):
            if self.expectedValue[i] == None:
                self.expectedValue[i] = sample[i]
            else:
                self.expectedValue[i] += (sample[i] - expectedValue[i]) / self.smoothFactor

        return self.expectedValue