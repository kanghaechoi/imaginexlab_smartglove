function [net,tr] = train_cnn(features, labels)
%% Convolution neural network training sequence

x = features';
y = labels';
trainFcn = 'trainscg';

hidden_layer_size = 100;
net = patternnet(hidden_layer_size);

net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
net.performFcn = 'crossentropy';

y_vector = ind2vec(y)

[net, tr] = train(net, x, y_vector);

end

