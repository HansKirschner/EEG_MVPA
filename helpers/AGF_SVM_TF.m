function [ Sensor_TF, Odata, TFdims, TFtimesMs, Sensor_TF2 ] = AGF_SVM_TF( indata, TF, PlotTime, Fdisp, Trials, Etotal, Odata)
%%
%performe time-frequency decomposition of the EEG data before running the SVM
%this replaces the time-domain data completely
UseEEGLAB=0;

%%
TFdims          = size(TF.bandfreq,2);
Padding = TF.padding;
if numel(Padding)==1
    Padding(2) = Padding;
elseif numel(Padding)>2
    disp('Error: Padding must be a vector of length 1 or 2.'); return;
end;

%find time-points with border at edges
if ~isempty(find(indata.times == PlotTime(1)-Padding(1)))
    TFt1 = find(indata.times == PlotTime(1)-Padding(1));
else
    disp('Warning: Data + padding at edge is larger than provided data, will use minimum of input data instead...')
    TFt1 = 1; 
end;
if ~isempty(find(indata.times == PlotTime(end)+Padding(2)))
    TFt2 = find(indata.times == PlotTime(end)+Padding(2));
else
    disp('Warning: Data + border at edge is larger than provided data, will use maximum of input data instead...')
    TFt2 = size(indata.data,2);
end;   
TFtimes   = TFt1:TFt2;
Borders   = [find(indata.times == PlotTime(1))-TFt1 TFt2-find(indata.times == PlotTime(end))];%this much border has been added
TFtimesMs = indata.times(TFtimes);
if ~isempty(TF.basetime)
    BaseInd = find(TFtimesMs == TF.basetime(1)):find(TFtimesMs == TF.basetime(2));
end;


switch(TF.space) % define if log or line spaced frequencies
    case 'log'
        frex = logspace(log10(TF.frequencies(1)),log10(TF.frequencies(2)),TF.stepnumber);
    case 'lin'
        frex = linspace(TF.frequencies(1),TF.frequencies(2),TF.stepnumber);
end;

%depending on chosen space and step size, not all values will be present: find closest match
for CM = 1 : TFdims
    [~,CloseInd1]       = min(abs(frex-TF.bandfreq{CM}(1)));
    TF.usedfreq{CM}(1)  = frex(CloseInd1);
    [~,CloseInd2]       = min(abs(frex-TF.bandfreq{CM}(2)));
    TF.usedfreq{CM}(2)  = frex(CloseInd2);
    CollapseIndex(CM,:) = [CloseInd1 CloseInd2];
end;


if length(TF.cyclenumber) == 1
    % the width of wavelets scales with the frequency (FWHM of gaussian)
    sncy = TF.cyclenumber./(2*pi.*frex);
else
    nCycles = logspace(log10(TF.cyclenumber(1)),log10(TF.cyclenumber(end)),length(frex));
    % the width of wavelets scales with the frequency (FWHM of gaussian)
    sncy    = nCycles./(2*pi.*frex);
end


% make wavelet
t = (indata.xmin-indata.xmax)/2 : 1/indata.srate : abs(indata.xmin-indata.xmax)/2;

for fi=1:length(frex)
    w(fi,:)=exp(2*1i*pi*frex(fi).*t).*exp(-t.^2./(2*sncy(fi)^2)); % sin(2*pi*f*t) IN Euler's formula (e^ik) * gaussian [(-t^2  / SD^s) *2]
end % plot(real(w(1,:)))

Sensor_TF       = nan(length(PlotTime),       length(Trials),size(Etotal,1), TF.stepnumber);  %4D time trial channel frequency
Dummy_TF        = nan(length(TFtimes),length(Trials),1,              TF.stepnumber);          %Dummy variable for each electrode to save ram
ExtractIndex    = find(ismember(TFtimesMs,PlotTime));
if TF.angle==1
    Sensor_TF2 = nan(length(PlotTime),       length(Trials),size(Etotal,1), TF.stepnumber*2);
    Dummy_TF2 = nan(length(TFtimes),length(Trials),1,TF.stepnumber+2);    %4D time trial channel angles (cos,sin)
else
    Dummy_TF2=nan;
end;
STrSmooth = '';
if TF.smoothing>0
    Tbef = floor(TF.smoothing/2);
    Tafter = ceil(TF.smoothing/2);
    STrSmooth=['Applying smoothing of ' num2str(Tbef) ' DP before and ' num2str(Tafter) ' DP after each TF time point.'];
    Dummy_TF_bu = Dummy_TF;
end;

                
%%

if Fdisp > 0
    disp(' ')
    disp('********************TF SETTINGS********************')
    disp('*******                                     *******')
    disp(['Frequency range from ' num2str(TF.frequencies(1)) ' to ' num2str(TF.frequencies(2)) ' Hz with ' num2str(TF.stepnumber) ' ' TF.space ' steps.'])
    if TF.angle == 1
        disp('Adding phase information as sine and cosine of the angle of each frequency to the data...')
    end;
    fprintf(STrSmooth);
    disp(' ')
end;
for chn = 1:length(Etotal)
    chnInd = Etotal(chn);       
    dataX = squeeze(indata.data(chnInd,TFtimes,Trials));
    if UseEEGLAB==1
        [ERSP_Arnaud,~,powbase,Atimes,freqs1,~,~,tfdata] = newtimef( dataX, 1750, [TFtimesMs(1) TFtimesMs(end)], 500, TF.cyclenumber, 'freqs', TF.frequencies, 'nfreqs', TF.stepnumber,...
            'freqscale', TF.space, 'baseline', NaN, 'plotitc', 'off', 'plotersp', 'off','scale','abs','trialbase', 'on', 'timesout', [-300:2:-100 270 280]);
        Sensor_TF(:,:,chn,:) = permute(abs(tfdata).^2, [2 3 1]);
        for fi = 1:size(tfdata,1)
            xbase                   = mean(Sensor_TF(1:find(Atimes==-100),:,chn,fi),1);
            Sensor_TF(:,:,chn,fi)  = (Sensor_TF(:,:,chn,fi)-repmat(xbase,size(Sensor_TF,1),1)) ./ mean(xbase);
        end;
    else
        if Fdisp > 0; fprintf('%s\t%s\t','Time-frequency transformation for: ',indata.chanlocs(chnInd).labels); end;
        for fi = 1:TF.stepnumber
            if rem(fi,5) == 0; fprintf('.'); end
            prepdata=fconv_tg(reshape(dataX,1,size(dataX,1)*size(dataX,2)),w(fi,:));    % convolve data with wavelet
            prepdata=prepdata((size(w,2)-1)/2:end-1-(size(w,2)-1)/2);                   % cut of 1/2 the length of the w from beg, and 1/2 from the end
            prepdata=reshape(prepdata,size(dataX,1),size(dataX,2));                     % reshape
            Dummy_TF(:,:,1,fi) = abs(prepdata).^2;                                      % Standard power
            if TF.angle==1 %also add phase information
                A = angle(prepdata);
                Dummy_TF2(:,:,1,fi) = sin(A);
                Dummy_TF2(:,:,1,fi+TF.stepnumber) = cos(A);
            end;
            %Use baseline correction for time-frequency data
            if ~isempty(TF.basetime)
                switch TF.basetype
%                     case 'percent'
%                        xbase                    = mean(Dummy_TF(BaseInd,:,1,fi),1);
%                        Dummy_TF(:,:,1,fi)       = (Dummy_TF(:,:,1,fi)-repmat(xbase,size(Dummy_TF,1),1)) ./ mean(xbase); %subtract baseline then divide
%                 end;
                case 'percent'
                    xbase                    = mean(Dummy_TF(BaseInd,:,1,fi),1);
                    Dummy_TF(:,:,1,fi)       = (Dummy_TF(:,:,1,fi)-repmat(xbase,size(Dummy_TF,1),1)) ./ mean(xbase); %subtract baseline then divide
                    STrBase = ['Basline info: subtracting single-trial baselines followed by normalization on the average.'];
                case 'subtract'
                    xbase                    = mean(Dummy_TF(BaseInd,:,1,fi),1);
                    Dummy_TF(:,:,1,fi)       = Dummy_TF(:,:,1,fi)-repmat(xbase,size(Dummy_TF,1),1);
                    STrBase = ['Basline info: and subtracting single-trial baselines.'];
                otherwise
                    STrBase = ['Basline info: and performing no baseline correction.'];
                end
            end; 
            %smooth data if so desired
            if TF.smoothing>0
                Dummy_TF_bu = Dummy_TF; %avoid that smoothing overwrites other datapoints
                for bc = Tbef+1 : length(TFtimes)-Tafter
                    Dummy_TF(bc,:,:,fi) = mean(Dummy_TF_bu(bc-Tbef:bc+Tafter-1,:,:,fi),1);
                end;
            end;
        end;
        Sensor_TF(:,:,chn,:) = Dummy_TF(ExtractIndex,:,1,:);
        if TF.angle==1 %also add phase information
            Sensor_TF2(:,:,chn,1:fi)               = Dummy_TF2(ExtractIndex,:,1,1:fi);        % sine of angle
            Sensor_TF2(:,:,chn,fi+1:fi*2)          = Dummy_TF2(ExtractIndex,:,1,fi+1:fi*2);   % cosine of angle 
        else
            Sensor_TF2 = nan;
        end;
        if Fdisp > 0; fprintf('\n'); end;
    end;
end

% TFtimesMs=Atimes;
%reduce data over frequency bands (if selected) 
if ~isempty(TF.bands)
    if Fdisp > 0; disp(['Collapsing over ' num2str(TFdims) ' frequency bands...']);end;
    for F =  1 : size(TF.bands,2)
        Sensor_TF_temp(:,:,:,F) = mean(Sensor_TF(:,:,:,CollapseIndex(F,1):CollapseIndex(F,2)),4);
    end;
    Sensor_TF = Sensor_TF_temp; 
    clear Sensor_TF_temp;
else
    TFdims = TF.stepnumber; 
end;

Odata.TF            = TF;
Odata.TF.Borders    = Borders;
Odata.TF.freq       = frex;

return;