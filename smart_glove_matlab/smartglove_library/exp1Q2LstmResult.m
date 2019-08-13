function [featuresTotalCell, labelsTotalCell] = exp1Q2LstmResult(A1, B1, ages, genderID)
%% Smart glove data read & feature extraction

    featuresMCell = {};
    featuresFCell = {};

    %Subjects in 20s
    for mIterationNum = 1 : 3
        for mSubjectNum = 1 : 10
            if mIterationNum == 1 && mSubjectNum == 1
                [featuresMData, labelsMData] = ...
                    exp1Q2FeatureExtractCell(B1, A1, ages, mIterationNum, genderID(1,1), mSubjectNum); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresMCell = {featuresMData};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsMCell = labelsMData;
                continue;
            else
                [featuresMData, labelsMData] = ...
                    exp1Q2FeatureExtractCell(B1, A1, ages, mIterationNum, genderID(1,1), mSubjectNum); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresMCell = [featuresMCell; {featuresMData}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsMCell = [labelsMCell; labelsMData];
            end
        end
    end

    for fIterationNum = 1 : 3
        for fSubjectNum = 1 : 8
            if fIterationNum == 1 && fSubjectNum == 1
                [featuresFData, labelsFData] = ...
                    exp1Q2FeatureExtractCell(B1, A1, ages, fIterationNum, genderID(1,2), fSubjectNum); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresFCell = {featuresFData};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsFCell = labelsFData;
                continue;
            else
                [featuresFData, labelsFData] = ...
                    exp1Q2FeatureExtractCell(B1, A1, ages, fIterationNum, genderID(1,2), fSubjectNum); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresFCell = [featuresFCell; {featuresFData}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsFCell = [labelsFCell; labelsFData];
            end
        end
    end
    
    %Subjects in total
    featuresTotalCell = [featuresMCell; featuresFCell]; 
    labelsTotalCell = [labelsMCell; labelsFCell];
    
end

