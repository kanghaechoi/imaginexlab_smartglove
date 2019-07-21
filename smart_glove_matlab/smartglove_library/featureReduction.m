function [reducedU, idx, varRetain] = featureReduction(inputMat)
%% Principle Component Analysis (PCA)
[numOfSample, numOfFeat] = size(inputMat);
varRetain = zeros(numOfFeat, 1);

sigma = (1 / numOfSample) * (inputMat' * inputMat); %Covariance
[U, S, V] = svd(sigma); %Singular Value Decomposition

for k = 1 : numOfFeat
    varRetain(k) = trace(S(1:k,1:k)) / trace(S); %Variance retain calculation
end

idx = find(varRetain > 0.99, 1); %Find variance error below 1%

reducedU = U(:,1:idx); % Feature reducing matrix
%reducedInput = reducedU' * inputMat'; %Reduced input matrix by PCA

end

