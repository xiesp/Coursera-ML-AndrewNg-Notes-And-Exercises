function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;

% mu的维度是 1- n,n是特征数量
mu = zeros(1, size(X, 2));

% 维度也是 1- n
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       


% 计算X每一列的平均值
mu = mean(X);

% subtract each feature's mean value
% 把X的每一列的值,都减去对应列的平均值
% 还可以这样写 :  X_norm = bsxfun(@minus, X, mu);
X_norm = X - mu;

% compute std and substract from fetures
% 计算方差
sigma = std(X);

% 再除以方差,注意 ./ 符号
% 或者这样写:  X_norm = bsxfun(@rdivide, X_norm, sigma); 
X_norm = X_norm ./ sigma;






% ============================================================

end
