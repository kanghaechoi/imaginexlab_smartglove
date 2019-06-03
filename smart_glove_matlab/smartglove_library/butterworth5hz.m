function [B1,A1] =butterworth5hz(SAMPLE_FREQ)
%% 5 Hz low-pass filter

NYQUIST_RATE = SAMPLE_FREQ/2; %Nyquist sampling rate
LPFCutOff = 5/NYQUIST_RATE;
ORDER = 5;
[B1,A1] = butter(ORDER, LPFCutOff, 'low');
end