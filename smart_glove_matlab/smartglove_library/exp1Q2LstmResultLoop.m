function [featuresTotalCell, labelsTotalCell] = exp1Q2LstmResultLoop(A1, B1, ages, genderID, numberOfFeatures)
%% Smart glove data read & feature extraction

    featuresM20Cell = {};
    featuresM40Cell = {};
    featuresM60Cell = {};
    featuresF20Cell = {};
    featuresF40Cell = {};
    featuresF60Cell = {};

    for m20IterationNum = 1 : 3
        for m20SubjectNum = 1 : 10
            if m20IterationNum == 1 && m20SubjectNum == 1
                [featuresM20Data, labelsM20Data] = ...
                    exp1Q2FeatureExtractLoop(B1, A1, ages(1,1), m20IterationNum, genderID(1,1), m20SubjectNum, numberOfFeatures); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresM20Cell = {featuresM20Data};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsM20Cell = labelsM20Data;
                continue;
            else
                [featuresM20Data, labelsM20Data] = ...
                    exp1Q2FeatureExtractLoop(B1, A1, ages(1,1), m20IterationNum, genderID(1,1), m20SubjectNum, numberOfFeatures); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresM20Cell = [featuresM20Cell; {featuresM20Data}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsM20Cell = [labelsM20Cell; labelsM20Data];
            end
        end
    end

    for m40IterationNum = 1 : 3
        for m40SubjectNum = 1 : 3
            if m40IterationNum == 1 && m40SubjectNum == 1
                [featuresM40Data, labelsM40Data] = ...
                    exp1Q2FeatureExtractLoop(B1, A1, ages(1,2), m40IterationNum, genderID(1,1), m40SubjectNum, numberOfFeatures); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresM40Cell = {featuresM40Data};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsM40Cell = labelsM40Data;
                continue;
            else
                [featuresM40Data, labelsM40Data] = ...
                    exp1Q2FeatureExtractLoop(B1, A1, ages(1,2), m40IterationNum, genderID(1,1), m40SubjectNum, numberOfFeatures); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresM40Cell = [featuresM40Cell; {featuresM40Data}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsM40Cell = [labelsM40Cell; labelsM40Data];
            end
        end
    end
    
    for m60IterationNum = 1 : 3
        for m60SubjectNum = 1 : 6
            if m60IterationNum == 1 && m60SubjectNum == 1
                [featuresM60Data, labelsM60Data] = ...
                    exp1Q2FeatureExtractLoop(B1, A1, ages(1,3), m60IterationNum, genderID(1,1), m60SubjectNum, numberOfFeatures); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresM60Cell = {featuresM60Data};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsM60Cell = labelsM60Data;
                continue;
            else
                [featuresM60Data, labelsM60Data] = ...
                    exp1Q2FeatureExtractLoop(B1, A1, ages(1,3), m60IterationNum, genderID(1,1), m60SubjectNum, numberOfFeatures); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresM60Cell = [featuresM60Cell; {featuresM60Data}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsM60Cell = [labelsM60Cell; labelsM60Data];
            end
        end
    end

    for f20IterationNum = 1 : 3
        for f20SubjectNum = 1 : 8
            if f20IterationNum == 1 && f20SubjectNum == 1
                [featuresF20Data, labelsF20Data] = ...
                    exp1Q2FeatureExtractLoop(B1, A1, ages(1,1), f20IterationNum, genderID(1,2), f20SubjectNum, numberOfFeatures); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresF20Cell = {featuresF20Data};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsF20Cell = labelsF20Data;
                continue;
            else
                [featuresF20Data, labelsF20Data] = ...
                    exp1Q2FeatureExtractLoop(B1, A1, ages(1,1), f20IterationNum, genderID(1,2), f20SubjectNum, numberOfFeatures); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresF20Cell = [featuresF20Cell; {featuresF20Data}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsF20Cell = [labelsF20Cell; labelsF20Data];
            end
        end
    end

    for f40IterationNum = 1 : 3
        for f40SubjectNum = 1 : 5
            if f40IterationNum == 1 && f40SubjectNum == 1
                [featuresF40Data, labelsF40Data] = ...
                    exp1Q2FeatureExtractLoop(B1, A1, ages(1,2), f40IterationNum, genderID(1,2), f40SubjectNum, numberOfFeatures); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresF40Cell = {featuresF40Data};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsF40Cell = labelsF40Data;
                continue;
            else
                [featuresF40Data, labelsF40Data] = ...
                    exp1Q2FeatureExtractLoop(B1, A1, ages(1,2), f40IterationNum, genderID(1,2), f40SubjectNum, numberOfFeatures); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresF40Cell = [featuresF40Cell; {featuresF40Data}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsF40Cell = [labelsF40Cell; labelsF40Data];
            end
        end
    end
    
    for f60IterationNum = 1 : 3
        for f60SubjectNum = 1 : 10
            if f60IterationNum == 1 && f60SubjectNum == 1
                [featuresF60Data, labelsF60Data] = ...
                    exp1Q2FeatureExtractLoop(B1, A1, ages(1,3), f60IterationNum, genderID(1,2), f60SubjectNum, numberOfFeatures); ...
                    %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresF60Cell = {featuresF60Data};
                %reducedFeatures20Cell = {reducedFeatures20Data};
                labelsF60Cell = labelsF60Data;
                continue;
            else
                [featuresF60Data, labelsF60Data] = ...
                    exp1Q2FeatureExtractLoop(B1, A1, ages(1,3), f60IterationNum, genderID(1,2), f60SubjectNum, numberOfFeatures); ...
                     %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
                featuresF60Cell = [featuresF60Cell; {featuresF60Data}];
                %reducedFeatures20Cell = [reducedFeatures20Cell; {reducedFeatures20Data}];
                labelsF60Cell = [labelsF60Cell; labelsF60Data];
            end
        end
    end
    
    %Subjects in total
    featuresTotalCell = [featuresM20Cell; featuresM40Cell; featuresM60Cell; ...
        featuresF20Cell; featuresF40Cell; featuresF60Cell]; 
    labelsTotalCell = [labelsM20Cell; labelsM40Cell; labelsM60Cell; ...
        labelsF20Cell; labelsF40Cell; labelsF60Cell];
    
end

