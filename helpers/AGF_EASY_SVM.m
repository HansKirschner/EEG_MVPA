function [ Accuracy, SaveToler, Weights, FullSVM, PT_Info ] = AGF_EASY_SVM( TrainingTrials, TestTrials, UseSensorInformation, UseY, IterNum, oldschool, Fdisp, ret_SVM, add_arguments,DoAvg,avgoverN)
%simple function that just calls matlab SVM functions with input generated by EEG_SVM.
%%

c_iteration=0;
if Fdisp
    fprintf(1,['Progress - Percent of analysis done:   ']);
end;
n_iterations    = size(TrainingTrials,1) * size(UseSensorInformation,1);
if size(UseSensorInformation,4)<=1
    Weights         = nan(size(UseSensorInformation,3), size(UseSensorInformation,1), size(TrainingTrials,1));
else
    Weights         = nan(size(UseSensorInformation,3)*size(UseSensorInformation,4), size(UseSensorInformation,1), size(TrainingTrials,1));
end;
Accuracy        = nan(size(UseSensorInformation,1), size(TrainingTrials,1));
if ret_SVM
    FullSVM     = cell(size(UseSensorInformation,1),1);
else
    FullSVM     = NaN;
end;
SaveToler       = Accuracy;

for n_fold = 1 : size(TrainingTrials,1) % loop over number of folds
    for K = 1 : size(UseSensorInformation,1)
        c_iteration = c_iteration + 1;
        SensorInformation = squeeze(UseSensorInformation(K,:,:,:));
        %Reshape data to an electrode by frequency array
        
        if size(SensorInformation,1)==1
            TrainX  = SensorInformation(:,TrainingTrials(n_fold,:))';
            TestX   = SensorInformation(:,TestTrials(n_fold,:))';
        else
            TrainX  = reshape(SensorInformation(TrainingTrials(n_fold,:),:,:),length(TrainingTrials(n_fold,:)),size(SensorInformation,2)*size(SensorInformation,3));
            TestX   = reshape(SensorInformation(TestTrials(n_fold,:),:,:),length(TestTrials(n_fold,:)),size(SensorInformation,2)*size(SensorInformation,3));
        end;
        
        if DoAvg > 0

            % ok, we want to run MVPA on ERP data. Let's do some averaging
            % in a rather clumsy way. But I couldn't find a vectorized
            % solution quickly...

            % here we go: find the trials to average over and pre-allocate
            % the variables

            % for Training set
            ix1Tr           = find(UseY(TrainingTrials(n_fold,:)) == 0);
            ix2Tr           = find(UseY(TrainingTrials(n_fold,:)) == 1);
            NrTr1           = floor(length(ix1Tr)/avgoverN);
            TrainXAVG       = nan(2*NrTr1,size(TrainX,2));
            TruthTraining   = nan(1,2*NrTr1);
            ix = 1;
            for ii = 1:NrTr1
                TrainXAVG(ii,:) = mean(TrainX(ix1Tr(ix:ix+avgoverN-1),:),1);
                if ii == NrTr1
                    TrainXAVG(ii,:) = mean(TrainX(ix1Tr(ix:end),:),1);
                end
                ix = 1 + avgoverN;
                TruthTraining(ii) = 0;
            end

            NrTr2 = floor(length(ix2Tr)/avgoverN);
            ix = 1;
            for ii = 1:NrTr2
                TrainXAVG(ii+NrTr1,:) = mean(TrainX(ix2Tr(ix:ix+avgoverN-1),:),1);
                if ii == NrTr2
                    TrainXAVG(ii+NrTr1,:) = mean(TrainX(ix2Tr(ix:end),:),1);
                end
                ix = 1 +avgoverN;
                TruthTraining(ii+NrTr1) = 1;
            end


            % for Test-set
            if DoAvg == 1
                ix1Te        = find(UseY(TestTrials(n_fold,:)) == 0);
                ix2Te        = find(UseY(TestTrials(n_fold,:)) == 1);
                NrTr1        = floor(length(ix1Te)/avgoverN);
                TestXAVG     = nan(2*NrTr1,size(TestX,2));
                TruthTesting = nan(1,2*NrTr1); 
                ix = 1;
                
                for ii = 1:NrTr1
                    TestXAVG(ii,:) = mean(TestX(ix1Te(ix:ix+avgoverN-1),:),1);
                    if ii == NrTr1
                        TestXAVG(ii,:) = mean(TestX(ix1Te(ix:end),:),1);
                    end
                    ix = 1 +avgoverN;
                    TruthTesting(ii) = 0;
                end

                NrTr2 = floor(length(ix2Te)/avgoverN);
                ix = 1;
                for ii = 1:NrTr2
                    TestXAVG(ii+NrTr1,:) = mean(TestX(ix2Te(ix:ix+avgoverN-1),:),1);
                    if ii == NrTr2
                        TestXAVG(ii+NrTr1,:) = mean(TestX(ix2Te(ix:end),:),1);
                    end
                    ix = 1 +avgoverN;
                    TruthTesting(ii+NrTr1) = 1;
                end

            elseif DoAvg == 2
                % use ERPs only for training
                TestXAVG     = TestX;
                TruthTesting = UseY(TestTrials(n_fold,:))';
            end
        end
        
        if oldschool == 1 %use old matlab functions svmtrain and svmpredict
            % No longer supported
            disp('Oldschool mode no longer supported...')
            return;
        elseif ~DoAvg %wonderful new world: use matlab fitcsvm and predict functions (much faster)
            if ~isempty(add_arguments)
                SVMmodel = eval(['fitcsvm(TrainX,UseY(TrainingTrials(n_fold,:)),''IterationLimit'', IterNum,' add_arguments ')']);
            else
                SVMmodel = fitcsvm(TrainX,UseY(TrainingTrials(n_fold,:)),'IterationLimit', IterNum);
            end;
            
            C = predict(SVMmodel,TestX);
            Accuracy(K,n_fold) = 1-sum(UseY(TestTrials(n_fold,:))~= C)/length(TestTrials(n_fold,:));
            PT_Info.model(K,n_fold,:) = C';
            PT_Info.Truth(K,n_fold,:) = UseY(TestTrials(n_fold,:));
            SaveToler(K,n_fold) = SVMmodel.NumIterations;
            Weights(:,K,n_fold) = SVMmodel.Beta'; % These are the weights assigned to each sensor
            if ret_SVM
                FullSVM(K) = {SVMmodel}; %should only be set, of no crossvalidation is run! Otherwise overwrites itself
            end;
            %         for c = 1 : 59
            %             W(c) = sum((SVMmodel.SupportVectors(:,c).*SVMmodel.Alpha))
            %         end;
            %         [x,y]=sort((W),'descend');
            %         indata.chanlocs(y(1:10)).labels
            if Fdisp %display progress
                do=floor((c_iteration / n_iterations) * 100);
                if do < 10
                    fprintf(2,'\b%d', do);
                elseif do < 100
                    fprintf(2,'\b\b%d', do);
                else
                    fprintf(2,'\b\b\b%d', do);
                    fprintf(2, '\n');
                end;
            end;
            
        elseif DoAvg %wonderful new world: use matlab fitcsvm and predict functions (much faster)
            if ~isempty(add_arguments)
                SVMmodel = eval(['fitcsvm(TrainXAVG,TruthTraining,''IterationLimit'', IterNum,' add_arguments ')']);
            else
                SVMmodel = fitcsvm(TrainXAVG,TruthTraining,'IterationLimit', IterNum);
                %SVMmodel = fitcecoc(TrainXAVG,TruthTraining, 'Coding','onevsall','Learners','SVM' );   %train support vector mahcine
                
            end;
            C = predict(SVMmodel,TestXAVG);
           
            Accuracy(K,n_fold) = 1-sum( TruthTesting ~= C')/size(TestXAVG,1);
            PT_Info.model(K,n_fold,:) = C';
            PT_Info.Truth(K,n_fold,:) = TruthTesting;
            %Accuracy(K,n_fold) = 1-sum(UseY(TestTrials(n_fold,:))~= C)/length(TestTrials(n_fold,:));
            SaveToler(K,n_fold) = SVMmodel.NumIterations;
           
            Weights(:,K,n_fold) = SVMmodel.Beta'; % These are the weights assigned to each sensor
            if ret_SVM
                FullSVM(K) = {SVMmodel}; %should only be set, of no crossvalidation is run! Otherwise overwrites itself
            end;
            %         for c = 1 : 59
            %             W(c) = sum((SVMmodel.SupportVectors(:,c).*SVMmodel.Alpha))
            %         end;
            %         [x,y]=sort((W),'descend');
            %         indata.chanlocs(y(1:10)).labels
            if Fdisp %display progress
                do=floor((c_iteration / n_iterations) * 100);
                if do < 10
                    fprintf(2,'\b%d', do);
                elseif do < 100
                    fprintf(2,'\b\b%d', do);
                else
                    fprintf(2,'\b\b\b%d', do);
                    fprintf(2, '\n');
                end;
            end;
        end;
    end;
end;
return;