clear all;
clc;
file = "D:\Develop\Dataset\clean_trainset_wav\p226_001.wav";
nfile = "D:\Develop\Dataset\noisy_trainset_wav\p226_001.wav";
info = audioinfo(file)
ninfo = audioinfo(nfile)
pcol=2;
prow=3;

%% Waveform
figure

[y,Fs] = audioread(file);
t = 0:seconds(1/Fs):seconds(info.Duration);
t = t(1:end-1);

%t = linspace(0, info.TotalSamples, size(s,1));
subplot(prow,pcol,1)
plot(t,y)
xlabel('Time')
title('Clear Audio Signal')

[ny,nFs] = audioread(nfile);
nt = 0:seconds(1/nFs):seconds(ninfo.Duration);
nt = nt(1:end-1);
subplot(prow,pcol,2)
plot(nt,ny)
xlabel('Time')
title('Noisy Audio Signal')

%% Short-time Fourier Transform
wlen = 256;

[s,f,t]=spectrogram(y,wlen,[],[],Fs); %Hamming window by default
%t = linspace(0, info.Duration, info.TotalSamples);
%f = linspace(0, Fs, size(s,2));
subplot(prow,pcol,3)
imagesc(t, f, 20*log10((abs(s))));xlabel('Time (s)'); ylabel('Freqency (Hz)');
%colorbar;
title('CAS using STFT')
%size(s)

[ns,nf,nt]=spectrogram(ny,wlen,[],[],nFs);
%nt = linspace(0, ninfo.TotalSamples, size(ns,1));
%nf = linspace(0, nFs, size(ns,2));
subplot(prow,pcol,4)
imagesc(nt, nf, 20*log10((abs(ns))));xlabel('Time (s)'); ylabel('Freqency (Hz)');
%colorbar;
title('NAS using STFT')

%% Wavelet Transform

subplot(prow,pcol,5)
[cfs,frq]=cwt(y,Fs);
tms = (0:numel(y)-1)/Fs;
surface(tms,frq,abs(cfs))
axis tight
shading flat
xlabel("Time (s)")
ylabel("Frequency (Hz)")
set(gca,"yscale","log")
title('CAS using CWT')

subplot(prow,pcol,6)
[ncfs,nfrq]=cwt(ny,nFs);
ntms = (0:numel(ny)-1)/nFs;
surface(ntms,nfrq,abs(ncfs))
axis tight
shading flat
xlabel("Time (s)")
ylabel("Frequency (Hz)")
set(gca,"yscale","log")
title('NAS using CWT')