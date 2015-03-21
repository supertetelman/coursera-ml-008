%ml-008 Exercise6
%Based off of sample code provided by coursera Machine Learning Course
%ml-008 taught by Andrew NG of Stanford
%@author Adam Tetelman 3/21/2015
function [C, sigma] = dataset3Params(X, y, Xval, yval)

%TODO auto populate options arrays via min/max/inc
%C_options = [0 1 2 4 8 16 32 64 128 256 512 1024 2048];
%sigma_options = [0 .1 .2 .4 .8 .16 .32 .64 1.28 2.56 5.12 10.24 20.48];
C_options = [.1 .5 1 5 10 50];
sigma_options = [.1 .5 1 5 10 50];

error = ones(length(C_options), length(sigma_options));

%TODO: Improve these x1,x2 values
x1 = [1 2 1]; x2 = [0 4 -1];

for i=1:length(C_options)
    for j=1:length(sigma_options)
        model = svmTrain(X,y,C_options(i),...
            @(x1, x2) gaussianKernel(x1, x2,sigma_options(j)),1e-3, 20);
        predictions = svmPredict(model, Xval);
        error(i,j) = mean(double(predictions ~= yval));
    end
end

%Return the sigma/C combination with the smallest error
[a x] = min(error);
[b d] = min(a);
C = C_options(x(d));
sigma = sigma_options(d);

end
