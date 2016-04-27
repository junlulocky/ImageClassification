%% PCML Project 2 - Object Detection
% AUTHORS:  - Jun Lu, SCIPER 254412
%           - Miguel Ferreira, SCIPER 259852

%% ADD LIBRARIES
addpath(genpath('DeepLearnToolbox-master'));

%% INITIALIZATION
clear; 
close all;
clc;

% Project Title
fprintf('\nWELCOME TO PCML PROJECT 2 MAIN\n');
pause;
clc
% Read Input Data
[X_train_hog, X_train_cnn, y_train]=readTrainData();
[X_test_hog, X_test_cnn]=readTestData();

%% SETUP MENU
fprintf('\nSETUP MENU');
% PCA Fatorization
prompt = '\n\nPress "1" for applying PCA fatorization and reduce data dimensionality. Press "0" otherwise.\n\n';
flagPCA= input(prompt);
if (flagPCA>1 || flagPCA<0)
    fprintf('\nInvalid Caracter. PCA Fatorization will not take place\n');
    pause(2);
end    
clc
% Outlier Removal
prompt = '\n\nPress "1" for remove outliers through K-Means. Press "0" otherwise.\n\n';
flagOutlier= input(prompt);
if (flagOutlier>1 || flagOutlier<0)
    fprintf('\nInvalid Caracter. Outlier removal will not take place\n');
    pause(2);
end
clc
% Solving Imbalanced Data Set
fprintf('\n\nPress "1" for solving imbalanced data set through Random Under Sampling.\n');
fprintf('Press "2" for solving imbalanced data set through Random Over Sampling.\n');
fprintf('Press "3" for solving imbalanced data set through Algorithm Over Sampling.\n');
prompt = 'Press "0" otherwise\n\n';
flagSolveImbalance= input(prompt);
switch flagSolveImbalance
    case 0
        flagSolveImbalance='None';
    case 1
        flagSolveImbalance='RandomUnderSampling';
    case 2
        flagSolveImbalance='RandomOverSampling';
    case 3
        flagSolveImbalance='AlgorithmicOverSampling';
    otherwise 
        flagSolveImbalance='None';
        fprintf('\nInvalid Caracter. Dataset will remain the same\n'); 
        pause(2);
end    
clc
% Infere Task to Perform
fprintf('\n\nPress "1" for experimenting ML Methods over the Training Set.\n');
fprintf('Press "2" for computing to Predict Error Distribution.\n');
prompt='Press "3" for computing to Predict Labels on Test Set.\n\n';
flagTask= input(prompt);
if (flagTask>3 || flagTask<0)
    fprintf('\nInvalid Caracter. Program will finish\n');
    pause(2);
    return;
end 
clc
if flagTask==1
% Select Features
fprintf('\n\nPress "1" for CNN Features\n');
prompt='Press "2" for HOG Features\n\n';
flagFeatures= input(prompt);
if (flagFeatures>2 || flagFeatures<0)
    fprintf('\nInvalid Caracter. Program will finish\n');
    pause(2);
    return;
end 
clc
% Select ML Method
fprintf('\n\nPress "1" for running Logistic Regression for Binary Classification\n');
fprintf('Press "2" for running SVM for Binary Classification\n');
fprintf('Press "3" for running Logistic Regression One vs Rest\n');
fprintf('Press "4" for running Multinomial Logistic Regression\n');
fprintf('Press "5" for running Multiclass SVM\n');
prompt= 'Press "6" for running Neural Network for Multiclass Classification\n\n';
flagMLMethod= input(prompt);
switch flagMLMethod
    case {1,3,4}
        fprintf('\nPress "1" for Polynomial Basis\n');
        prompt= 'Press "2" for Gaussian Basis\n\n';
        flagBasis=input(prompt);
        switch flagBasis
            case 1
                flagBasis='PolyBasis';
                fprintf('\nPress "1" for Degree 1\n');
                fprintf('Press "2" for Degree 2\n');
                prompt='Press "3" for Degree 3\n\n';              
                Deg=input(prompt);
                if Deg>3 || Deg <1
                    fprintf('\nInvalid Caracter. Programm will finish\n'); 
                    pause(2);
                    return;
                end
            case 2
               flagBasis='GaussianBasis';

        end    
    case {2,5}       
        fprintf('\n\nPress "1" for Linear Kernel\n');
        fprintf('Press "2" for Polynomial Kernel\n');
        prompt= 'Press "3" for RBF Kernel\n\n';
        flagKernel=input(prompt);
        switch flagKernel 
            case 1
                flagKernel='LinearKernel';
            case 2    
                flagKernel='PolyKernel';
                fprintf('\n\nPress "1" for Degree 1\n');
                fprintf('Press "2" for Degree 2\n');
                prompt='Press "3" for Degree 3\n';
                Deg=input(prompt);
                if Deg>3 || Deg <1
                    fprintf('\nInvalid Caracter. Programm will finish\n'); 
                    pause(2);
                    return;
                end
            case 3
                flagKernel='rbfKernel';
        end
    case {6}
    otherwise 
        fprintf('\nInvalid Caracter. Programm will finish\n');   
        pause(2);
        return;

end
clc


%% Select Features
switch flagFeatures
    case 1
        X=X_train_cnn;
        y=y_train;
    case 2
        X=X_train_hog;
        y=y_train;
end

end
%% Feature Engineering
switch flagTask
    case 1
        [y,X,~]=featureEngineering(y,X,flagPCA,flagOutlier,flagSolveImbalance,flagFeatures);
    case 2
        [y,X,~]=featureEngineering(y_train,X_train_cnn,1,0,'None',1);
    case 3
        [yTr,XTr,V]=featureEngineering(y_train,X_train_cnn,1,0,'None',1);
end

%% Training Applying ML Methods
switch flagTask
% Apply Methods 
    case 1       
        switch flagMLMethod
            case 1
                [berErrorTr,berErrorTe]=learnLogisticRegression(y,X,flagBasis,Deg);
            case 2
                [berErrorTr,berErrorTe]=learnSVM(y_train,X_train,flagKernel,Deg);    
            case 3
                [berErrorTr,berErrorTe]=learnLogisticRegressionOnevsAll(y,X,flagBasis,Deg);
            case 4
                [berErrorTr,berErrorTe]=learnMultinomLogisticRegression(y,X,flagBasis,Deg);
            case 5
                [berErrorTr,berErrorTe]=learnMultiClassSVM(y,X,flagKernel,Deg);        
            case 6
                [berErrorTr,berErrorTe]=learnNeuralNetworks(y,X);      
        end

%% Predict Test Error Distribution
    case 2
        learnDistribTestError(y,X);
%% Compute Predictions for Test Set based on the Best Model
    case 3
        [yPredBinClass,yPredMultiClass]=learnBestModel(yTr,XTr,X_test_cnn,V);
    otherwise    
end
