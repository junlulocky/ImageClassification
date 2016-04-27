function [X_test_hog, X_test_cnn] = readTestData()
% Read Project Input Test Data
% Input: -
% Output: Test data in Hog Features;
%         Test data in Cnn Features;

load('test.mat');
X_test_hog=test.X_hog;
X_test_cnn=test.X_cnn;

end

