function [ranks, weights] = relieffResultExp2(A1, B1, maleID, femaleID, ...
    ages, gestureID)
%% Smart glove data read & feature extraction & rms

    %Data from Subject 1
    featuresIndividual = [];
    labelsIndividual = [];
    rmsIndividual = [];
    
    for nIndividual = 1 : 5
        [featuresIndividualData, labelsIndividualData, rmsIndividualData] ... 
            = featureExtractionExp2(B1, A1, ages, nIndividual, maleID, ... 
            01, gestureID(:,4), 1); ...
            %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
        featuresIndividual = [featuresIndividual; featuresIndividualData];
        labelsIndividual = [labelsIndividual; labelsIndividualData];
        rmsIndividual = [rmsIndividual; rmsIndividualData];
    end
    
    for nIndividual = 1 : 5
        [featuresIndividualData, labelsIndividualData, rmsIndividualData] ... 
            = featureExtractionExp2(B1, A1, ages, nIndividual, maleID, ... 
            01, gestureID(:,5), 1); ...
            %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
        featuresIndividual = [featuresIndividual; featuresIndividualData];
        labelsIndividual = [labelsIndividual; labelsIndividualData];
        rmsIndividual = [rmsIndividual; rmsIndividualData];
    end
    
        for nIndividual = 1 : 5
        [featuresIndividualData, labelsIndividualData, rmsIndividualData] ... 
            = featureExtractionExp2(B1, A1, ages, nIndividual, maleID, ... 
            01, gestureID(:,6), 1); ...
            %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
        featuresIndividual = [featuresIndividual; featuresIndividualData];
        labelsIndividual = [labelsIndividual; labelsIndividualData];
        rmsIndividual = [rmsIndividual; rmsIndividualData];
    end

    featuresMOthers = [];
    labelsMOthers = [];
    rmsMOthers = [];
    
    for nMOthers = 1 : 5
        for mSubjectNum = 2 : 5
            [featuresMOthersData, labelsMOthersData, rmsMOthersData] ... 
                = featureExtractionExp2(B1, A1, ages, nMOthers, maleID, ... 
                mSubjectNum, gestureID(:,4), 0); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresMOthers = [featuresMOthers; featuresMOthersData];
            labelsMOthers = [labelsMOthers; labelsMOthersData];
            rmsMOthers = [rmsMOthers; rmsMOthersData];
        end
    end
    
    for nMOthers = 1 : 5
        for mSubjectNum = 2 : 5
            [featuresMOthersData, labelsMOthersData, rmsMOthersData] ... 
                = featureExtractionExp2(B1, A1, ages, nMOthers, maleID, ... 
                mSubjectNum, gestureID(:,5), 0); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresMOthers = [featuresMOthers; featuresMOthersData];
            labelsMOthers = [labelsMOthers; labelsMOthersData];
            rmsMOthers = [rmsMOthers; rmsMOthersData];
        end
    end 
    
    for nMOthers = 1 : 5
        for mSubjectNum = 2 : 5
            [featuresMOthersData, labelsMOthersData, rmsMOthersData] ... 
                = featureExtractionExp2(B1, A1, ages, nMOthers, maleID, ... 
                mSubjectNum, gestureID(:,6), 0); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresMOthers = [featuresMOthers; featuresMOthersData];
            labelsMOthers = [labelsMOthers; labelsMOthersData];
            rmsMOthers = [rmsMOthers; rmsMOthersData];
        end
    end 
    
    featuresFOthers = [];
    labelsFOthers = [];
    rmsFOthers = [];
    
    for nFOthers = 1 : 5
        for fSubjectNum = 1 : 3
            [featuresFOthersData, labelsFOthersData, rmsFOthersData] ... 
                = featureExtractionExp2(B1, A1, ages, nFOthers, femaleID, ... 
                fSubjectNum, gestureID(:,4), 0); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresFOthers = [featuresFOthers; featuresFOthersData];
            labelsFOthers = [labelsFOthers; labelsFOthersData];
            rmsFOthers = [rmsFOthers; rmsFOthersData];
        end
    end   
    
    for nFOthers = 1 : 5
        for fSubjectNum = 1 : 3
            [featuresFOthersData, labelsFOthersData, rmsFOthersData] ... 
                = featureExtractionExp2(B1, A1, ages, nFOthers, femaleID, ... 
                fSubjectNum, gestureID(:,5), 0); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresFOthers = [featuresFOthers; featuresFOthersData];
            labelsFOthers = [labelsFOthers; labelsFOthersData];
            rmsFOthers = [rmsFOthers; rmsFOthersData];
        end
    end
    
    for nFOthers = 1 : 5
        for fSubjectNum = 1 : 3
            [featuresFOthersData, labelsFOthersData, rmsFOthersData] ... 
                = featureExtractionExp2(B1, A1, ages, nFOthers, femaleID, ... 
                fSubjectNum, gestureID(:,6), 0); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresFOthers = [featuresFOthers; featuresFOthersData];
            labelsFOthers = [labelsFOthers; labelsFOthersData];
            rmsFOthers = [rmsFOthers; rmsFOthersData];
        end
    end
    
    rmsTotal = [rmsIndividual; rmsMOthers; rmsFOthers];
    labelsTotal = [labelsIndividual; labelsMOthers; labelsFOthers];
    
    [ranks, weights] = relieff(rmsTotal, labelsTotal, 3, ...
        'method', 'classification');
    
    bar(weights(ranks))
    xlabel('Predictor rank');
    ylabel('Predictor important weight');

end

