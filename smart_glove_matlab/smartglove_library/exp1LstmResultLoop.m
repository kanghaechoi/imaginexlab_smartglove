function [featuresTotalCell, labelsTotalCell] = exp1LstmResultLoop(A1, B1, ages, fileCounts, genderID, numberOfFeatures)
%% Smart glove data read & feature extraction

    featuresM20Cell = {};
    featuresF20Cell = {};
    featuresM40Cell = {};
    featuresF40Cell = {};
    featuresM60Cell = {};
    featuresF60Cell = {};

    %Subjects in 20s
    for nM20 = 1 : fileCounts(:,1)
        if nM20 == 1
            [featuresM20Data, labelsM20Data] = ...
                exp1FeatureExtractLoop(B1, A1, ages(:,1), nM20, genderID(1,1), numberOfFeatures); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresM20Cell = {featuresM20Data};
            %reducedFeatures20Cell = {reducedFeatures20Data};
            labelsM20Cell = labelsM20Data;
            continue;
        else
            [featuresM20Data, labelsM20Data] = ...
                exp1FeatureExtractLoop(B1, A1, ages(:,1), nM20, genderID(1,1), numberOfFeatures); ...
                 %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresM20Cell = [featuresM20Cell; {featuresM20Data}];
            %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
            labelsM20Cell = [labelsM20Cell; labelsM20Data];
        end
    end

    for nF20 = 1 : fileCounts(:,2)
        if nF20 == 1
            [featuresF20Data, labelsF20Data] = ...
                exp1FeatureExtractLoop(B1, A1, ages(:,1), nF20, genderID(1,2), numberOfFeatures); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresF20Cell = {featuresF20Data};
            %reducedFeatures20Cell = {reducedFeatures20Data};
            labelsF20Cell = labelsF20Data;
            continue;
        else
            [featuresF20Data, labelsF20Data] = ...
                exp1FeatureExtractLoop(B1, A1, ages(:,1), nF20, genderID(1,2), numberOfFeatures); ...
                 %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresF20Cell = [featuresF20Cell; {featuresF20Data}];
            %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
            labelsF20Cell = [labelsF20Cell; labelsF20Data];
        end
    end
    
    %Subjects in 40s
    for nM40 = 1 : fileCounts(:,3)
        if nM40 == 1
            [featuresM40Data, labelsM40Data] = ...
                exp1FeatureExtractLoop(B1, A1, ages(:,3), nM40, genderID(1,1), numberOfFeatures); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresM40Cell = {featuresM40Data};
            %reducedFeatures40Cell = {reducedFeatures40Data};
            labelsM40Cell = labelsM40Data;
            continue;
        else
            [featuresM40Data, labelsM40Data] = ...
                exp1FeatureExtractLoop(B1, A1, ages(:,3), nM40, genderID(1,1), numberOfFeatures); ...
                 %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresM40Cell = [featuresM40Cell; {featuresM40Data}];
            %reducedFeatures40Cell = [reducedFeatures40Cell; {reducedFeatures40Data}];
            labelsM40Cell = [labelsM40Cell; labelsM40Data];
        end
    end
    
    for nF40 = 1 : fileCounts(:,4)
        if nF40 == 1
            [featuresF40Data, labelsF40Data] = ...
                exp1FeatureExtractLoop(B1, A1, ages(:,4), nF40, genderID(1,2), numberOfFeatures); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresF40Cell = {featuresF40Data};
            %reducedFeatures40Cell = {reducedFeatures40Data};
            labelsF40Cell = labelsF40Data;
            continue;
        else
            [featuresF40Data, labelsF40Data] = ...
                exp1FeatureExtractLoop(B1, A1, ages(:,4), nF40, genderID(1,2), numberOfFeatures); ...
                 %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresF40Cell = [featuresF40Cell; {featuresF40Data}];
            %reducedFeatures40Cell = [reducedFeatures40Cell; {reducedFeatures40Data}];
            labelsF40Cell = [labelsF40Cell; labelsF40Data];
        end
    end
    
    %Subjects in 60s
    for nM60 = 1 : fileCounts(:,5)
        if nM60 == 1
            [featuresM60Data, labelsM60Data] = ...
                exp1FeatureExtractLoop(B1, A1, ages(:,5), nM60, genderID(1,1), numberOfFeatures); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresM60Cell = {featuresM60Data};
            %reducedFeatures60Cell = {reducedFeatures60Data};
            labelsM60Cell = labelsM60Data;
            continue;
        else
            [featuresM60Data, labelsM60Data] = ...
                exp1FeatureExtractLoop(B1, A1, ages(:,5), nM60, genderID(1,1), numberOfFeatures); ...
                 %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresM60Cell = [featuresM60Cell; {featuresM60Data}];
            %reducedFeatures60Cell = [reducedFeatures60Cell; {reducedFeatures60Data}];
            labelsM60Cell = [labelsM60Cell; labelsM60Data];
        end
    end
    
    for nF60 = 1 : fileCounts(:,6)
        if nF60 == 1
            [featuresF60Data, labelsF60Data] = ...
                exp1FeatureExtractLoop(B1, A1, ages(:,6), nF60, genderID(1,2), numberOfFeatures); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresF60Cell = {featuresF60Data};
            %reducedFeatures60Cell = {reducedFeatures60Data};
            labelsF60Cell = labelsF60Data;
            continue;
        else
            [featuresF60Data, labelsF60Data] = ...
                exp1FeatureExtractLoop(B1, A1, ages(:,6), nF60, genderID(1,2), numberOfFeatures); ...
                 %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresF60Cell = [featuresF60Cell; {featuresF60Data}];
            %reducedFeatures60Cell = [reducedFeatures60Cell; {reducedFeatures60Data}];
            labelsF60Cell = [labelsF60Cell; labelsF60Data];
        end
    end
    
    %Subjects in total
    featuresTotalCell = [featuresM20Cell; featuresF20Cell;...
        featuresM40Cell; featuresF40Cell;...
        featuresM60Cell; featuresF60Cell]; %Features from 20s + Features from 60s
    %reducedFeaturesTotalCell = [reducedFeatures20Cell; ...
    %    reducedFeatures40Cell; ...
    %    reducedFeatures60Cell];
    labelsTotalCell = [labelsM20Cell; labelsF20Cell;... 
        labelsM40Cell; labelsF40Cell;...
        labelsM60Cell; labelsF60Cell];
end

