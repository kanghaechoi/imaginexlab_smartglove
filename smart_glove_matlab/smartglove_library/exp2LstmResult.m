function [featuresTotalCell, labelsTotalCell] = exp2LstmResult(A1, B1, ages, genderID, gestureID)
%% Smart glove data read & feature extraction
    featuresMCell = {};
    featuresFCell = {};

    %Subjects in 20s
    for mIterationNum = 1 : 5
        for mSubjectNum = 1 : 5
            if mIterationNum == 1 && mSubjectNum == 1
                [featuresMData, labelsMData] = ...
                    exp2FeatureExtractCell(B1, A1, ages, mIterationNum, genderID(1,1), mSubjectNum, gestureID(1,4)); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresMCell = {featuresMData};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsMCell = labelsMData;
                continue;
            else
                [featuresMData, labelsMData] = ...
                    exp2FeatureExtractCell(B1, A1, ages, mIterationNum, genderID(1,1), mSubjectNum, gestureID(1,4)); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresMCell = [featuresMCell; {featuresMData}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsMCell = [labelsMCell; labelsMData];
            end
        end
    end

    for fIterationNum = 1 : 5
        for fSubjectNum = 1 : 3
            if fIterationNum == 1 && fSubjectNum == 1
                [featuresFData, labelsFData] = ...
                    exp2FeatureExtractCell(B1, A1, ages, fIterationNum, genderID(1,2), fSubjectNum, gestureID(1,4)); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresFCell = {featuresFData};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsFCell = labelsFData;
                continue;
            else
                [featuresFData, labelsFData] = ...
                    exp2FeatureExtractCell(B1, A1, ages, fIterationNum, genderID(1,2), fSubjectNum, gestureID(1,4)); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresFCell = [featuresFCell; {featuresFData}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsFCell = [labelsFCell; labelsFData];
            end
        end
    end


    %Subjects in total
    featuresTotalCell = [featuresMCell; featuresFCell]; %Features from 20s + Features from 60s
    %reducedFeaturesTotalCell = [reducedFeatures20Cell; ...
    %    reducedFeatures40Cell; ...
    %    reducedFeatures60Cell];
    labelsTotalCell = [labelsMCell; labelsFCell];
    
end

