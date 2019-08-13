function [ranks, weights] = exp1Q2RelieffResult(A1, B1, maleID, femaleID, ages)
%% Smart glove data read & feature extraction & rms

    featuresMIndividual = [];
    labelsMIndividual = [];
    rmsMIndividual = [];
    
    for nMIndividual = 1 : 3
        for mSubjectNum = 1 : 10
            [featuresMIndividualData, labelsMIndividualData, rmsMIndividualData] ... 
                = exp1Q2FeatureExtract(B1, A1, ages, nMIndividual, maleID, ... 
                mSubjectNum); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresMIndividual = [featuresMIndividual; featuresMIndividualData];
            labelsMIndividual = [labelsMIndividual; labelsMIndividualData];
            rmsMIndividual = [rmsMIndividual; rmsMIndividualData];
        end
    end
    
    featuresFIndividual = [];
    labelsFIndividual = [];
    rmsFIndividual = [];
    
    for nFIndividual = 1 : 3
        for fSubjectNum = 1 : 8
            [featuresFIndividualData, labelsFIndividualData, rmsFIndividualData] ... 
                = exp1Q2FeatureExtract(B1, A1, ages, nFIndividual, femaleID, ... 
                fSubjectNum); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresFIndividual = [featuresFIndividual; featuresFIndividualData];
            labelsFIndividual = [labelsFIndividual; labelsFIndividualData];
            rmsFIndividual = [rmsFIndividual; rmsFIndividualData];
        end
    end
    
    rmsTotal = [rmsMIndividual; rmsFIndividual];
    labelsTotal = [labelsMIndividual; labelsFIndividual];
    
    [ranks, weights] = relieff(rmsTotal, labelsTotal, 19, ...
        'method', 'classification');
    
    bar(weights(ranks))
    xlabel('Predictor rank');
    ylabel('Predictor important weight');

end

