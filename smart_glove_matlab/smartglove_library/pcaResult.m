function [pcaMat, numOfK, varRetain] = pcaResult(ages, fileCounts)
%% Smart glove data read & feature extraction
    %Subjects in 20s
    for nM20 = 1 : fileCounts(:,1)
        if nM20 == 1
            [featuresM20Data] = featureExtractionPca(ages(:,1), nM20); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresM20 = featuresM20Data;
            continue;
        else
            [featuresM20Data] = ...
                featureExtractionPca(ages(:,1), nM20); ...
                 %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresM20 = [featuresM20; featuresM20Data];
        end
    end
    
    for nF20 = 1 : fileCounts(:,2)
        if nF20 == 1
            [featuresF20Data] = featureExtractionPca(ages(:,2), nF20); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresF20 = featuresF20Data;
            continue;
        else
            [featuresF20Data] = ...
                featureExtractionPca(ages(:,2), nF20); ...
                 %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresF20 = [featuresF20; featuresF20Data];
        end
    end
    
    %Subjects in 40s
    for nM40 = 1 : fileCounts(:,3)
        if nM40 == 1
            [featuresM40Data] = featureExtractionPca2(age40, nM40); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresM40 = featuresM40Data;
            continue;
        else
            [featuresM40Data] = ...
                featureExtractionPca(age40, nM40); ...
                 %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresM40 = [featuresM40; featuresM40Data];
        end
    end
    
    for nM40 = 1 : fileCounts(:,3)
        if nM40 == 1
            [featuresM40Data] = featureExtractionPca2(ages(:,3), nM40); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresM40 = featuresM40Data;
            continue;
        else
            [featuresM40Data] = ...
                featureExtractionPca(ages(:,3), nM40); ...
                 %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresM40 = [featuresM40; featuresM40Data];
        end
    end
    
    for nF40 = 1 : fileCounts(:,4)
        if nF40 == 1
            [featuresF40Data] = featureExtractionPca2(ages(:,4), nF40); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresF40 = featuresF40Data;
            continue;
        else
            [featuresF40Data] = ...
                featureExtractionPca(ages(:,4), nF40); ...
                 %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresF40 = [featuresF40; featuresF40Data];
        end
    end
        
    %Subjects in 60s
    for nM60 = 1 : fileCounts(:,5)
        if nM60 == 1
            [featuresM60Data] = featureExtractionPca2(ages(:,5), nM60); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresM60 = featuresM60Data;
            continue;
        else
            [featuresM60Data] = ...
                featureExtractionPca(ages(:,5), nM60); ...
                 %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresM60 = [featuresM60; featuresM60Data];
        end
    end
    
    for nF60 = 1 : fileCounts(:,6)
        if nF60 == 1
            [featuresF60Data] = featureExtractionPca2(ages(:,6), nF60); ...
                %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresF60 = featuresF60Data;
            continue;
        else
            [featuresM60Data] = ...
                featureExtractionPca(ages(:,6), nF60); ...
                 %[features] = feature_extraction(b1, a1, age, file_count): Feature extraction
            featuresF60 = [featuresF60; featuresF60Data];
        end
    end
    
    %Subjects in total
    featuresTotal = [featuresM20; featuresF20; featuresM40; featuresF40; ...
        featuresM60; featuresF60]; %Features from 20s + Features from 60s
    scaledFeaturesTotal = featureScale(featuresTotal);
    
%% Principle Component Analysis
    [pcaMat, numOfK, varRetain] = featureReduction(featuresTotal);

    fprintf("The ideal number of K = %d\n\n", numOfK);
    
end

