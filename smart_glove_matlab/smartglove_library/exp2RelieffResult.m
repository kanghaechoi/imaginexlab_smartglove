function [ranks, weights] = exp2RelieffResult(A1, B1, maleID, femaleID, ages, gestureID)
%% Smart glove data read & feature extraction & rms

    %Data from Subject 1
    featuresMIndividual = [];
    labelsMIndividual = [];
    rmsMIndividual = [];
    
    for nMIndividual = 1 : 5
        for mSubjectNum = 1 : 5
            [featuresMIndividualData, labelsMIndividualData, rmsMIndividualData] ... 
                = exp2FeatureExtract(B1, A1, ages, nMIndividual, maleID, ... 
                mSubjectNum, gestureID(1,6)); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresMIndividual = [featuresMIndividual; featuresMIndividualData];
            labelsMIndividual = [labelsMIndividual; labelsMIndividualData];
            rmsMIndividual = [rmsMIndividual; rmsMIndividualData];
        end
    end
    
    featuresFIndividual = [];
    labelsFIndividual = [];
    rmsFIndividual = [];
    
    for nFIndividual = 1 : 5
        for fSubjectNum = 1 : 3
            [featuresFIndividualData, labelsFIndividualData, rmsFIndividualData] ... 
                = exp2FeatureExtract(B1, A1, ages, nFIndividual, femaleID, ... 
                fSubjectNum, gestureID(1,6)); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresFIndividual = [featuresFIndividual; featuresFIndividualData];
            labelsFIndividual = [labelsFIndividual; labelsFIndividualData];
            rmsFIndividual = [rmsFIndividual; rmsFIndividualData];
        end
    end
    
    rmsTotal = [rmsMIndividual; rmsFIndividual];
    labelsTotal = [labelsMIndividual; labelsFIndividual];
    
    [ranks, weights] = relieff(rmsTotal, labelsTotal, 9, ...
        'method', 'classification');
    
    bar(weights(ranks))
    xlabel('Predictor rank');
    ylabel('Predictor important weight');

end

