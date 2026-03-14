#!/usr/bin/python
#import sys

import numpy as np
import torch

class rotd_gpu(object):
    """
    Class to calculate spectral acceleration for horizontal and vertical components.
    Code are modified from pyrotd, please check the link: "https://github.com/arkottke/pyrotd"
    Main changes:
    1. enable multiple waveforms processing, pyrotd can only process one waveform at a time
    2. shift from numpy to torch, and use GPU for acceleration
    """
    def __init__(self, device):
        self.device = device
    
    def calc_oscillator_resp(self,
        freq,
        fourier_amp,
        osc_damping,
        osc_freq,
        max_freq_ratio=5.0,
        peak_resp_only=False,
        osc_type="psa",
    ):
        """Compute the time series response of an oscillator.

        Parameters
        ----------
        freq : array_like
            frequency of the Fourier acceleration spectrum [Hz]
        fourier_amp : array_like
            Fourier acceleration spectrum [g-sec]
        osc_damping : float
            damping of the oscillator [decimal]
        osc_freq : float
            frequency of the oscillator [Hz]
        max_freq_ratio : float, default=5
            minimum required ratio between the oscillator frequency and
            then maximum frequency of the time series. It is recommended that this
            value be 5.
        peak_resp_only : bool, default=False
            If only the peak response is returned.
        osc_type : str, default='psa'
            type of response. Options are:
                'sd': spectral displacement
                'sv': spectral velocity
                'sa': spectral acceleration
                'psv': psuedo-spectral velocity
                'psa': psuedo-spectral acceleration
        Returns
        -------
        response : :class:`numpy.ndarray` or float
            time series response of the oscillator
        """

        # osc_freq is a single value
        ang_freq = (2 * torch.pi * freq).to(self.device)
        osc_ang_freq = (2 * torch.pi * osc_freq).to(self.device)

        # Single-degree of freedom transfer function
        h = 1 / (
            ang_freq ** 2.0
            - osc_ang_freq ** 2
            - 2.0j * osc_damping * osc_ang_freq * ang_freq
        ).to(self.device)
        if osc_type == "sd":
            pass
        elif osc_type == "sv":
            h *= 1.0j * ang_freq
        elif osc_type == "sa":
            h *= 1 + (1.0j * ang_freq) ** 2
        elif osc_type == "psa":
            h *= -(osc_ang_freq ** 2)
        elif osc_type == "psv":
            h *= -osc_ang_freq
        else:
            raise RuntimeError

        # Adjust the maximum frequency considered. The maximum frequency is 5
        # times the oscillator frequency. This provides that at the oscillator
        # frequency there are at least tenth samples per wavelength.
        n = fourier_amp.shape[1]
        #m = max(n, int(max_freq_ratio * osc_freq / freq[1]))
        m = n # remove the restriction of max_freq_ratio (it recommends that it shouldn't larger than 50/5)
        scale = float(m) / float(n)

        # Scale factor is applied to correct the amplitude of the motion for the
        # change in number of points
        resp = scale * torch.fft.irfft(fourier_amp * h, 2 * (m - 1))

        if peak_resp_only:
            resp = torch.abs(resp).max(-1)[0]

        return resp


    def calc_rotated_percentiles(self,accels, angles, percentiles=None):
        """Compute the response spectrum for a time series.

        Parameters
        ----------
        accels : list of array_like
            pair of acceleration time series
        angles : array_like
            angles to which to compute the rotated time series
        percentiles : array_like or None
            percentiles to return

        Returns
        -------
        rotated_resp : :class:`np.recarray`
            Percentiles of the rotated response. Records have keys:
            'percentile', 'spec_accel', and 'angle'.
        """
        percentiles = (
            0.5 if percentiles is None else percentiles
        )
        angles = torch.arange(0, 180, step=1).to(self.device)

        # Compute rotated time series
        radians = torch.deg2rad(angles).to(self.device)
        coeffs = torch.zeros(len(angles), 2).to(self.device)
        coeffs[:,0] = torch.cos(radians)
        coeffs[:,1] = torch.sin(radians)

        #coeffs = torch.tensor(np.c_[np.cos(radians), np.sin(radians)]).to('cuda')

        rotated_time_series = torch.einsum('ij,bjk->bik', coeffs, accels)
        #rotated_time_series = np.dot(coeffs, accels)
        # Sort this array based on the response
        peak_responses = torch.abs(rotated_time_series).max(dim=-1)[0]
        #print("Peak responses shape:", peak_responses.shape)
        #rotated = torch.cat((angles.repeat(peak_responses.shape[0], 1).unsqueeze(1), peak_responses.unsqueeze(1)), dim=1)

        #rotated = np.rec.fromarrays([angles, peak_responses], names="angle,peak_resp")
        #rotated.sort(dim=-1)

        #angles = rotated[:,0,:]
        #peak_respones = rotated[:,1,:]
        peak_responses = peak_responses.sort(dim=-1)[0]

        # Get the peak response at the requested percentiles
        p_peak_resps = torch.quantile(peak_responses, percentiles, interpolation='linear', dim=1, keepdim=False)
        # Can only return the orientations for the minimum and maximum value as the
        # orientation is not unique (i.e., two values correspond to the 50%
        # percentile).
        """
        p_angles = np.select(
            [np.isclose(percentiles, 0), np.isclose(percentiles, 100), True],
            [rotated.angle[0], rotated.angle[-1], np.nan],
        )
        return np.rec.fromarrays(
            [percentiles, p_peak_resps, p_angles], names="percentile,spec_accel,angle"
        )    

        """
        return p_peak_resps


    def calc_rotated_oscillator_resp(
        self,angles, percentiles, freqs, fourier_amps, osc_damping, osc_freq, max_freq_ratio=5.0
    ):
        """Compute the percentiles of response of a rotated oscillator.

        Parameters
        ----------
        percentiles : array_like
            percentiles to return.
        angles : array_like
            angles to which to compute the rotated time series.
        freq : array_like
            frequency of the Fourier acceleration spectrum [Hz]
        fourier_amps : [array_like, array_like]
            pair of Fourier acceleration spectrum [g-sec]
        osc_damping : float
            damping of the oscillator [decimal]
        osc_freq : float
            frequency of the oscillator [Hz]
        max_freq_ratio : float, default=5
            minimum required ratio between the oscillator frequency and
            then maximum frequency of the time series. It is recommended that this
            value be 5.
        peak_resp_only : bool, default=False
            If only the peak response is returned.

        Returns
        -------
        response : :class:`numpy.ndarray` or float
            time series response of the oscillator
        """

        # Compute the oscillator responses

        #100, 6000 -> 100, 1, 6000


        osc_ts = torch.zeros(fourier_amps[0].shape[0], 2, (fourier_amps[0].shape[1]-1)*2).to(self.device)
        for i in range(2):
            osc_ts[:,i,:] = self.calc_oscillator_resp(
                    freqs,
                    fourier_amps[i],
                    osc_damping,
                    osc_freq,
                    max_freq_ratio=max_freq_ratio,
                    peak_resp_only=False,
                )

        # Compute the rotated values of the oscillator response
        # osc_ts.shape [N, 2, 6000]
        rotated_percentiles = self.calc_rotated_percentiles(osc_ts, angles, percentiles)

        # Stack all of the results
        #return [(osc_freq,) + rp.tolist() for rp in rotated_percentiles]
        return rotated_percentiles 


    def calc_spec_accels(
        self,time_step, accel_ts, osc_freqs, osc_damping=0.05, max_freq_ratio=5
    ):
        """Compute the psuedo-spectral accelerations.

        Parameters
        ----------
        time_step : float
            time step of the time series [s]
        accel_ts : array_like
            acceleration time series [g]
        osc_freqs : array_like
            natural frequency of the oscillators [Hz]
        osc_damping : float
            damping of the oscillator [decimal]. Default of 0.05 (i.e., 5%)
        max_freq_ratio : float, default=5
            minimum required ratio between the oscillator frequency and
            then maximum frequency of the time series. It is recommended that this
            value be 5.

        Returns
        -------
        resp_spec : :class:`np.recarray`
            computed pseudo-spectral acceleration [g]. Records have keys:
            'osc_freq', and 'spec_accel'
        """
        # The input accel_ts shape be 2D [batch_size, length]
        fourier_amp = torch.fft.rfft(accel_ts)

        freq = torch.linspace(0, 1.0 / (2 * time_step), steps=fourier_amp.shape[1])


            # Single process
        spec_accels = torch.zeros(len(osc_freqs), len(accel_ts))

        for i in range(len(osc_freqs)):
            spec_accels[i] = self.calc_oscillator_resp(
                freq,
                fourier_amp,
                osc_damping,
                osc_freqs[i],
                max_freq_ratio=max_freq_ratio,
                peak_resp_only=True,
            )

        spec_accels = spec_accels.permute(1, 0)

        return osc_freqs, spec_accels

    #np.rec.fromarrays([osc_freqs, spec_accels], names="osc_freq,spec_accel")


    def calc_rotated_spec_accels(
        self,
        time_step,
        accel_a,
        accel_b,
        osc_freqs,
        osc_damping=0.05,
        percentiles=None,
        angles=None,
        max_freq_ratio=5,
    ):
        """
        Compute the rotated psuedo-spectral accelerations.
        """
        percentiles = 0.5 if percentiles is None else percentiles
        angles = torch.arange(0, 180, step=1)

        assert accel_a.shape[1] == accel_b.shape[1], "Time series not equal lengths!"

        # Compute the Fourier amplitude spectra
        fourier_amps = [torch.fft.rfft(accel_a), torch.fft.rfft(accel_b)]
        freqs = torch.linspace(0, 1.0 / (2 * time_step), steps=fourier_amps[0].shape[1]).to(self.device)


            # Single process

        groups = torch.zeros(len(osc_freqs), len(accel_a)).to(self.device)


        for i in range(len(osc_freqs)):

            groups[i] = self.calc_rotated_oscillator_resp(
                angles,
                percentiles,
                freqs,
                fourier_amps,
                osc_damping,
                osc_freqs[i],
                max_freq_ratio=max_freq_ratio,
            )

        groups = groups.permute(1, 0)
        #records = [g for group in groups for g in group]

        #return rotated_resp
        return osc_freqs, groups