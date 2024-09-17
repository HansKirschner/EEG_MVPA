function [y]=fconv_tg(x, h)
% modified version of fconv from Stephen G. McGovern
% Script by Theo Gründler. 
Ly=length(x)+length(h)-1;   
Ly2=pow2(nextpow2(Ly));     % Find smallest power of 2 that is > Ly
X=fft(x, Ly2);              % Fast Fourier transform
H=fft(h, Ly2);              % Fast Fourier transform
Y=X.*H;                     
y=ifft(Y, Ly2);             % Inverse fast Fourier transform
y=y(1:1:Ly);                % Take just the first N elements
end