function [net,tr] = trainCnn(features, labels)
%% Convolution neural network training sequence

x = features';
y = labels';
trainFcn = 'trainscg';

hiddenLayerSize = 100;
net = patternnet(hiddenLayerSize);

net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
net.performFcn = 'crossentropy';

yVector = ind2vec(y);

[net, tr] = train(net, x, yVector);

end

