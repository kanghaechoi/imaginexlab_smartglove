function [ranks, weights] = exp1Q2RelieffResult(A1, B1, maleID, femaleID, ages)
%% Smart glove data read & feature extraction & rms

    featuresM20Individual = [];
    labelsM20Individual = [];
    rmsM20Individual = [];
    
    for nM20Individual = 1 : 3
        for m20SubjectNum = 1 : 10
            [featuresM20IndividualData, labelsM20IndividualData, rmsM20IndividualData] ... 
                = exp1Q2FeatureExtract(B1, A1, ages(1,1), nM20Individual, maleID, ... 
                m20SubjectNum); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresM20Individual = [featuresM20Individual; featuresM20IndividualData];
            labelsM20Individual = [labelsM20Individual; labelsM20IndividualData];
            rmsM20Individual = [rmsM20Individual; rmsM20IndividualData];
        end
    end
    
    featuresM40Individual = [];
    labelsM40Individual = [];
    rmsM40Individual = [];
    
    for nM40Individual = 1 : 3
        for m40SubjectNum = 1 : 3
            [featuresM40IndividualData, labelsM40IndividualData, rmsM40IndividualData] ... 
                = exp1Q2FeatureExtract(B1, A1, ages(1,2), nM40Individual, maleID, ... 
                m40SubjectNum); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresM40Individual = [featuresM40Individual; featuresM40IndividualData];
            labelsM40Individual = [labelsM40Individual; labelsM40IndividualData];
            rmsM40Individual = [rmsM40Individual; rmsM40IndividualData];
        end
    end
    
    featuresM60Individual = [];
    labelsM60Individual = [];
    rmsM60Individual = [];
    
    for nM60Individual = 1 : 3
        for m60SubjectNum = 1 : 6
            [featuresM60IndividualData, labelsM60IndividualData, rmsM60IndividualData] ... 
                = exp1Q2FeatureExtract(B1, A1, ages(1,3), nM60Individual, maleID, ... 
                m60SubjectNum); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresM60Individual = [featuresM60Individual; featuresM60IndividualData];
            labelsM60Individual = [labelsM60Individual; labelsM60IndividualData];
            rmsM60Individual = [rmsM60Individual; rmsM60IndividualData];
        end
    end
    
    featuresF20Individual = [];
    labelsF20Individual = [];
    rmsF20Individual = [];
    
    for nF20Individual = 1 : 3
        for f20SubjectNum = 1 : 8
            [featuresF20IndividualData, labelsF20IndividualData, rmsF20IndividualData] ... 
                = exp1Q2FeatureExtract(B1, A1, ages(1,1), nF20Individual, femaleID, ... 
                f20SubjectNum); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresF20Individual = [featuresF20Individual; featuresF20IndividualData];
            labelsF20Individual = [labelsF20Individual; labelsF20IndividualData];
            rmsF20Individual = [rmsF20Individual; rmsF20IndividualData];
        end
    end
    
    featuresF40Individual = [];
    labelsF40Individual = [];
    rmsF40Individual = [];
    
    for nF40Individual = 1 : 3
        for f40SubjectNum = 1 : 5
            [featuresF40IndividualData, labelsF40IndividualData, rmsF40IndividualData] ... 
                = exp1Q2FeatureExtract(B1, A1, ages(1,2), nF40Individual, femaleID, ... 
                f40SubjectNum); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresF40Individual = [featuresF40Individual; featuresF40IndividualData];
            labelsF40Individual = [labelsF40Individual; labelsF40IndividualData];
            rmsF40Individual = [rmsF40Individual; rmsF40IndividualData];
        end
    end
    
    featuresF60Individual = [];
    labelsF60Individual = [];
    rmsF60Individual = [];
    
    for nF60Individual = 1 : 3
        for f60SubjectNum = 1 : 10
            [featuresF60IndividualData, labelsF60IndividualData, rmsF60IndividualData] ... 
                = exp1Q2FeatureExtract(B1, A1, ages(1,3), nF60Individual, femaleID, ... 
                f60SubjectNum); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresF60Individual = [featuresF60Individual; featuresF60IndividualData];
            labelsF60Individual = [labelsF60Individual; labelsF60IndividualData];
            rmsF60Individual = [rmsF60Individual; rmsF60IndividualData];
        end
    end
    
    rmsTotal = [rmsM20Individual; rmsM40Individual; rmsM60Individual;...
        rmsF20Individual; rmsF40Individual; rmsF60Individual];
    labelsTotal = [labelsM20Individual; labelsM40Individual; labelsM60Individual; ... 
        labelsF20Individual; labelsF40Individual; labelsF60Individual];
    
    [ranks, weights] = relieff(rmsTotal, labelsTotal, 43, ...
        'method', 'classification');
    
    bar(weights(ranks))
    xlabel('Predictor rank');
    ylabel('Predictor important weight');

end

