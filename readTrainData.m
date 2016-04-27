function [X_train_hog, X_train_cnn, y_train] = readTrainData()
% Read Project Input Train Data
% Input: -
% Output: Training data in Hog Features;
%         Training data in Cnn Features;
%         Training data Labels;

load('train.mat');
X_train_hog=train.X_hog;
X_train_cnn=train.X_cnn;
y_train=train.y;

end

