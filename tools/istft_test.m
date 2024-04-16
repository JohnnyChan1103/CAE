clear all;
clc;
file = "D:\Develop\Dataset\clean_trainset_wav\p226_001.wav";
info = audioinfo(file)
pcol=2;
prow=2;

%% Waveform
figure

[y,Fs] = audioread(file);
t = 0:seconds(1/Fs):seconds(info.Duration);
t = t(1:end-1);

%t = linspace(0, info.TotalSamples, size(s,1));
subplot(prow,pcol,1)
plot(t,y)
xlabel('Time')
title('Time-Domain Audio Signal')

%% Short-time Fourier Transform
wlen = 256;
[ss,f,t]=spectrogram(y,wlen,[],[],Fs); %Hamming window by default
subplot(prow,pcol,4)
imagesc(t, f, 20*log10((abs(ss))));xlabel('Time (s)'); ylabel('Freqency (Hz)');
%colorbar;
title('Spectrogram')

s=stft(y,Fs,'Window',hamming(wlen),'OverlapLength',128);
size(s)
subplot(prow,pcol,2)
imagesc(t, f, 20*log10((abs(s))));xlabel('Time (s)'); ylabel('Freqency (Hz)');
%colorbar;
title('STFT')
%abs(s(1:8,1:8))

%% Inverse Short-time Fourier Transform
[y,t]=istft(s,Fs,'Window',hamming(wlen),'OverlapLength',128);
subplot(prow,pcol,3)
plot(t,y)
xlabel('Time')
title('Reconstructed Time-Domain Audio Signal')