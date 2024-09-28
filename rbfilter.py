#!/usr/bin/env python
# -*- coding utf-8 -*-
import numpy as np
from scipy import signal

class rb_filter:
    def __init__(self, r_p, r_m, c, width=100, delta_t=1):
        self.t_w = np.arange(-width, 0, delta_t)
        self.weight = (np.sin(self.t_w/r_p)/self.t_w - np.sin(self.t_w/r_m)/self.t_w) * np.sin(self.t_w/c - np.pi)/(self.t_w/c - np.pi)
        self.t_w = np.append(self.t_w, 0)
        self.weight = np.append(self.weight, 0.)
        self.delta_t = delta_t
        self.width = width

        w, h = signal.freqz(self.weight) # w: angular frequency, h: response
        self.weight = self.weight / np.max(np.abs(h)) # rescale self.weight as the max(h)=0dB

    def calc(self, data):
        data_flt = np.zeros_like(data)
        data_flt[:] = np.nan
        for i in range(self.width, data.size):
            data_flt[i] = np.sum(data[i-self.width:i+1]*self.weight)
        return data_flt

    def timeweight(self):
        return self.t_w, self.weight

    def frequency(self):
        w, h = signal.freqz(self.weight)
        f = w/(2.*np.pi*self.delta_t) # frequency
        return f, h