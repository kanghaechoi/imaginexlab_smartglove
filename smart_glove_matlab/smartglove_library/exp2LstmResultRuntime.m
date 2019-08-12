function [featuresTotalCell, labelsTotalCell] = exp2LstmResult(A1, B1, ages, fileCounts, genderID, NumberFeatures)
%% Smart glove data read & feature extraction
    featuresMG1Cell = {};
    featuresFG1Cell = {};
    featuresMG2Cell = {};
    featuresFG2Cell = {};
    featuresMG3Cell = {};
    featuresFG3Cell = {};
    featuresMG4Cell = {};
    featuresFG4Cell = {};
    featuresMG5Cell = {};
    featuresFG5Cell = {};
    featuresMG6Cell = {};
    featuresFG6Cell = {};
    gesture1ID = [1 2 3 4 5 6];
    maleID = 1;
    femaleID = 2;
    %Subjects in 20s
    for nM20 = 1 : 5
        for iter = 1 : 5
            if nM20 == 1 && iter == 1
                [featuresMG1Data, labelsMG1Data] = ...
                    featureExtractionToCell2Runtime(B1, A1, ages(:,1), nM20, maleID, iter, gesture1ID(:,1), NumberFeatures); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresMG1Cell = {featuresMG1Data};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsMG1Cell = labelsMG1Data;
                continue;
            else
                [featuresMG1Data, labelsMG1Data] = ...
                    featureExtractionToCell2Runtime(B1, A1, ages(:,1), nM20, maleID, iter, gesture1ID(:,1), NumberFeatures); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresMG1Cell = [featuresMG1Cell; {featuresMG1Data}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsMG1Cell = [labelsMG1Cell; labelsMG1Data];
            end
        end
    end

    for nF20 = 1 : 5
        for iter = 1 : 3
            if nF20 == 1 && iter == 1
                [featuresFG1Data, labelsFG1Data] = ...
                    featureExtractionToCell2Runtime(B1, A1, ages(:,1), nF20, femaleID, iter, gesture1ID(:,1), NumberFeatures); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresFG1Cell = {featuresFG1Data};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsFG1Cell = labelsFG1Data;
                continue;
            else
                [featuresFG1Data, labelsFG1Data] = ...
                    featureExtractionToCell2Runtime(B1, A1, ages(:,1), nF20, femaleID, iter, gesture1ID(:,1), NumberFeatures); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresFG1Cell = [featuresFG1Cell; {featuresFG1Data}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsFG1Cell = [labelsFG1Cell; labelsFG1Data];
            end
        end
    end

     for nM20 = 1 : 5
        for iter = 1 : 5
            if nM20 == 1 && iter == 1
                [featuresMG2Data, labelsMG2Data] = ...
                    featureExtractionToCell2Runtime(B1, A1, ages(:,1), nM20, maleID, iter, gesture1ID(:,2), NumberFeatures); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresMG2Cell = {featuresMG2Data};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsMG2Cell = labelsMG2Data;
                continue;
            else
                [featuresMG2Data, labelsMG2Data] = ...
                    featureExtractionToCell2Runtime(B1, A1, ages(:,1), nM20, maleID, iter, gesture1ID(:,2), NumberFeatures); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresMG2Cell = [featuresMG2Cell; {featuresMG2Data}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsMG2Cell = [labelsMG2Cell; labelsMG2Data];
            end
        end
    end

    for nF20 = 1 : 5
        for iter = 1 : 3
            if nF20 == 1 && iter == 1
                [featuresFG2Data, labelsFG2Data] = ...
                    featureExtractionToCell2Runtime(B1, A1, ages(:,1), nF20, femaleID, iter, gesture1ID(:,2), NumberFeatures); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresFG2Cell = {featuresFG2Data};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsFG2Cell = labelsFG2Data;
                continue;
            else
                [featuresFG2Data, labelsFG2Data] = ...
                    featureExtractionToCell2Runtime(B1, A1, ages(:,1), nF20, femaleID, iter, gesture1ID(:,2), NumberFeatures); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresFG2Cell = [featuresFG2Cell; {featuresFG2Data}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsFG2Cell = [labelsFG2Cell; labelsFG2Data];
            end
        end
    end

    for nM20 = 1 : 5
        for iter = 1 : 5
            if nM20 == 1 && iter == 1
                [featuresMG3Data, labelsMG3Data] = ...
                    featureExtractionToCell2Runtime(B1, A1, ages(:,1), nM20, maleID, iter, gesture1ID(:,3), NumberFeatures); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresMG3Cell = {featuresMG3Data};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsMG3Cell = labelsMG3Data;
                continue;
            else
                [featuresMG3Data, labelsMG3Data] = ...
                    featureExtractionToCell2Runtime(B1, A1, ages(:,1), nM20, maleID, iter, gesture1ID(:,3), NumberFeatures); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresMG3Cell = [featuresMG3Cell; {featuresMG3Data}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsMG3Cell = [labelsMG3Cell; labelsMG3Data];
            end
        end
    end

    for nF20 = 1 : 5
        for iter = 1 : 3
            if nF20 == 1 && iter == 1
                [featuresFG3Data, labelsFG3Data] = ...
                    featureExtractionToCell2Runtime(B1, A1, ages(:,1), nF20, femaleID, iter, gesture1ID(:,3), NumberFeatures); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresFG3Cell = {featuresFG3Data};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsFG3Cell = labelsFG3Data;
                continue;
            else
                [featuresFG3Data, labelsFG3Data] = ...
                    featureExtractionToCell2Runtime(B1, A1, ages(:,1), nF20, femaleID, iter, gesture1ID(:,3), NumberFeatures); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresFG3Cell = [featuresFG3Cell; {featuresFG3Data}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsFG3Cell = [labelsFG3Cell; labelsFG3Data];
            end
        end
    end

    for nM20 = 1 : 5
        for iter = 1 : 5
            if nM20 == 1 && iter == 1
                [featuresMG4Data, labelsMG4Data] = ...
                    featureExtractionToCell2Runtime(B1, A1, ages(:,1), nM20, maleID, iter, gesture1ID(:,4), NumberFeatures); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresMG4Cell = {featuresMG4Data};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsMG4Cell = labelsMG4Data;
                continue;
            else
                [featuresMG4Data, labelsMG4Data] = ...
                    featureExtractionToCell2Runtime(B1, A1, ages(:,1), nM20, maleID, iter, gesture1ID(:,4), NumberFeatures); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresMG4Cell = [featuresMG4Cell; {featuresMG4Data}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsMG4Cell = [labelsMG4Cell; labelsMG4Data];
            end
        end
    end

    for nF20 = 1 : 5
        for iter = 1 : 3
            if nF20 == 1 && iter == 1
                [featuresFG4Data, labelsFG4Data] = ...
                    featureExtractionToCell2Runtime(B1, A1, ages(:,1), nF20, femaleID, iter, gesture1ID(:,4), NumberFeatures); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresFG4Cell = {featuresFG4Data};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsFG4Cell = labelsFG4Data;
                continue;
            else
                [featuresFG4Data, labelsFG4Data] = ...
                    featureExtractionToCell2Runtime(B1, A1, ages(:,1), nF20, femaleID, iter, gesture1ID(:,4), NumberFeatures); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresFG4Cell = [featuresFG4Cell; {featuresFG4Data}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsFG4Cell = [labelsFG4Cell; labelsFG4Data];
            end
        end
    end

    for nM20 = 1 : 5
        for iter = 1 : 5
            if nM20 == 1 && iter == 1
                [featuresMG5Data, labelsMG5Data] = ...
                    featureExtractionToCell2Runtime(B1, A1, ages(:,1), nM20, maleID, iter, gesture1ID(:,5), NumberFeatures); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresMG5Cell = {featuresMG5Data};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsMG5Cell = labelsMG5Data;
                continue;
            else
                [featuresMG5Data, labelsMG5Data] = ...
                    featureExtractionToCell2Runtime(B1, A1, ages(:,1), nM20, maleID, iter, gesture1ID(:,5), NumberFeatures); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresMG5Cell = [featuresMG5Cell; {featuresMG5Data}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsMG5Cell = [labelsMG5Cell; labelsMG5Data];
            end
        end
    end

    for nF20 = 1 : 5
        for iter = 1 : 3
            if nF20 == 1 && iter == 1
                [featuresFG5Data, labelsFG5Data] = ...
                    featureExtractionToCell2Runtime(B1, A1, ages(:,1), nF20, femaleID, iter, gesture1ID(:,5), NumberFeatures); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresFG5Cell = {featuresFG5Data};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsFG5Cell = labelsFG5Data;
                continue;
            else
                [featuresFG5Data, labelsFG5Data] = ...
                    featureExtractionToCell2Runtime(B1, A1, ages(:,1), nF20, femaleID, iter, gesture1ID(:,5), NumberFeatures); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresFG5Cell = [featuresFG5Cell; {featuresFG5Data}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsFG5Cell = [labelsFG5Cell; labelsFG5Data];
            end
        end
    end

    for nM20 = 1 : 5
        for iter = 1 : 5
            if nM20 == 1 && iter == 1
                [featuresMG6Data, labelsMG6Data] = ...
                    featureExtractionToCell2Runtime(B1, A1, ages(:,1), nM20, maleID, iter, gesture1ID(:,6), NumberFeatures); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresMG6Cell = {featuresMG6Data};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsMG6Cell = labelsMG6Data;
                continue;
            else
                [featuresMG6Data, labelsMG6Data] = ...
                    featureExtractionToCell2Runtime(B1, A1, ages(:,1), nM20, maleID, iter, gesture1ID(:,6), NumberFeatures); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresMG6Cell = [featuresMG6Cell; {featuresMG6Data}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsMG6Cell = [labelsMG6Cell; labelsMG6Data];
            end
        end
    end

    for nF20 = 1 : 5
        for iter = 1 : 3 
            if nF20 == 1 && iter == 1
                [featuresFG6Data, labelsFG6Data] = ...
                    featureExtractionToCell2Runtime(B1, A1, ages(:,1), nF20, femaleID, iter, gesture1ID(:,6), NumberFeatures); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresFG6Cell = {featuresFG6Data};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsFG6Cell = labelsFG6Data;
                continue;
            else
                [featuresFG6Data, labelsFG6Data] = ...
                    featureExtractionToCell2Runtime(B1, A1, ages(:,1), nF20, femaleID, iter, gesture1ID(:,6), NumberFeatures); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresFG6Cell = [featuresFG6Cell; {featuresFG6Data}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsFG6Cell = [labelsFG6Cell; labelsFG6Data];
            end
        end
    end

    %Subjects in total
    featuresTotalCell = [featuresMG1Cell; featuresFG1Cell;...
        featuresMG2Cell; featuresFG2Cell;...
        featuresMG3Cell; featuresFG3Cell;...
        featuresMG4Cell; featuresFG4Cell;...
        featuresMG5Cell; featuresFG5Cell;...
        featuresMG6Cell; featuresFG6Cell]; %Features from 20s + Features from 60s
    %reducedFeaturesTotalCell = [reducedFeatures20Cell; ...
    %    reducedFeatures40Cell; ...
    %    reducedFeatures60Cell];
    labelsTotalCell = [labelsMG1Cell; labelsFG1Cell;... 
        labelsMG2Cell; labelsFG2Cell;...
        labelsMG3Cell; labelsFG3Cell;...
        labelsMG4Cell; labelsFG4Cell;...
        labelsMG5Cell; labelsFG5Cell;...
        labelsMG6Cell; labelsFG6Cell];
    
end

