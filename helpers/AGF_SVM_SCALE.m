function [ indata, SaveScale, DispSTR ] = AGF_SVM_SCALE( indata, scale_dim, scaling, DispSTR)
%%
%Scale data
if scale_dim == 0 %scale over full dataset
    %scale datasets
    if scaling == 1 %-1 to 1
        T = indata - min(min(min(min(indata))));  %subtract lowest value
        indata = T./(max(max(max(max(T))))/2)-1;             %divide by maximum
        SaveScale = [min(min(min(min(indata))))-1 max(max(max(max(T))))/2]; 
        DispSTR = [DispSTR 'data is scaled from -1 to 1 across all trials and electrodes.\n'];
    elseif scaling == 2 %z score dataset
        [T, mu, sigma] = zscore(reshape(indata, 1, size(indata,2)*size(indata,3)*size(indata,1)*size(indata,4))); % has to be transformed to 1D, otherwise operates column wise
        SaveScale = [mu sigma];
        indata = reshape(T, size(indata,1),size(indata,2),size(indata,3),size(indata,4)); % restore 4D array...
        DispSTR = [DispSTR 'data is is z-scored across all trials and electrodes.\n'];
    else
        disp('Error: Scaling argument set to unknown value!')
        return;
    end;
elseif scale_dim == 1  %scale each channel separately
    if scaling == 1    %set min max values to [-1 1]
        DispSTR = [DispSTR 'data is scaled from -1 to 1 for every electrode separately.\n'];
        for chanl = 1 : size(indata,3)
            Cwork = indata(:,:,chanl,:);
            T = Cwork - min(min(min(min(Cwork))));
            indata(:,:,chanl,:) = T./(max(max(max(max(T))))/2)-1;
            SaveScale(chanl,1:2) = [min(min(min(min(Cwork))))-1 max(max(max(max(T))))/2];
        end;
    elseif scaling == 2 %z score channels
        DispSTR = [DispSTR 'data is z-scored for every electrode separately.\n'];
        for chanl = 1 : size(indata,3)
            [T, mu, sigma] = zscore(reshape(indata(:,:,chanl,:), 1, size(indata,2)*size(indata,1)*size(indata,4))); % has to be transformed to 1D, otherwise operates column wise
            SaveScale(chanl,1:2) = [mu sigma];
            indata(:,:,chanl,:) = reshape(T, size(indata,1),size(indata,2),size(indata,4)); % restore 4D array...
        end;
    else
        disp('Error: scale_dim argument set to unknown value!')
        return;
    end;
end;


return;