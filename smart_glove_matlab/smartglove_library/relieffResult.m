function [ranks, weights] = relieffResult(A1, B1, maleID, femaleID, ...
    ages, fileCounts)
%% Smart glove data read & feature extraction & rms

    %Subjects in 20s
    featuresM20 = [];
    labelsM20 = [];
    rmsM20 = [];
    
    for nM20 = 1 : fileCounts(:,1)
        [featuresM20Data, labelsM20Data, rmsM20Data] ... 
            = featureExtraction(B1, A1, ages(:,1), nM20, maleID); ...
            %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
        featuresM20 = [featuresM20; featuresM20Data];
        labelsM20 = [labelsM20; labelsM20Data];
        rmsM20 = [rmsM20; rmsM20Data];
    end

    featuresF20 = [];
    labelsF20 = [];
    rmsF20 = [];
    
    for nF20 = 1 : fileCounts(:,2)
        [featuresF20Data, labelsF20Data, rmsF20Data] ... 
            = featureExtraction(B1, A1, ages(:,2), nF20, femaleID); ...
            %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
        featuresF20 = [featuresF20; featuresF20Data];
        labelsF20 = [labelsF20; labelsF20Data];
        rmsF20 = [rmsF20; rmsF20Data];
    end    

    %Subjects in 40s
    featuresM40 = [];
    labelsM40 = [];
    rmsM40 = [];    

    for nM40 = 1 : fileCounts(:,3)
        [featuresM40Data, labelsM40Data, rmsM40Data] ...
            = featureExtraction(B1, A1, ages(:,3), nM40, maleID); ...
            %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
        featuresM40 = [featuresM40; featuresM40Data];
        labelsM40 = [labelsM40; labelsM40Data];
        rmsM40 = [rmsM40; rmsM40Data];
    end

    featuresF40 = [];
    labelsF40 = [];
    rmsF40 = [];  
    
    for nF40 = 1 : fileCounts(:,4)
        [featuresF40Data, labelsF40Data, rmsF40Data] ...
            = featureExtraction(B1, A1, ages(:,4), nF40, femaleID); ...
            %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
        featuresF40 = [featuresF40; featuresF40Data];
        labelsF40 = [labelsF40; labelsF40Data];
        rmsF40 = [rmsF40; rmsF40Data];
    end    
    
    %Subjects in 60s
    featuresM60 = [];
    labelsM60 = [];
    rmsM60 = [];    
    
    for nM60 = 1 : fileCounts(:,5)
        [featuresM60Data, labelsM60Data, rmsM60Data] ...
            = featureExtraction(B1, A1, ages(:,5), nM60, maleID); ...
            %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
        featuresM60 = [featuresM60; featuresM60Data];
        labelsM60 = [labelsM60; labelsM60Data];
        rmsM60 = [rmsM60; rmsM60Data];
    end

    featuresF60 = [];
    labelsF60 = [];
    rmsF60 = [];    
    
    for nF60 = 1 : fileCounts(:,6)
        [featuresF60Data, labelsF60Data, rmsF60Data] ...
            = featureExtraction(B1, A1, ages(:,6), nF60, femaleID); ...
            %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
        featuresF60 = [featuresF60; featuresF60Data];
        labelsF60 = [labelsF60; labelsF60Data];
        rmsF60 = [rmsF60; rmsF60Data];
    end
    
    rmsTotal = [rmsM20; rmsF20; rmsM40; rmsF40; rmsM60; rmsF60];
    labelsTotal = [labelsM20; labelsF20; labelsM40; labelsF40; labelsM60; labelsF60];
    
    [ranks, weights] = relieff(rmsTotal, labelsTotal, 6, ...
        'method', 'classification');
    
    bar(weights(ranks))
    xlabel('Predictor rank');
    ylabel('Predictor important weight');

end

