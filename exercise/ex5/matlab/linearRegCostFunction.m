function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% X在ex5.m已经添加过一列1了,所以下面X.sahpe = [m 2]
% y.shape = [m 1],theta.shape = [2 1]
% lambda被设置为1
devider = 2.0 * m;
% because this will be reused more than once,cached to improve performance
y_pred = X * theta;
J = 1.0 / devider * sum(sum((y_pred - y) .^2)) + lambda / devider * sum(sum(theta(2:end,:).^2));

% 首先设置theta(1) = 0,避免下面分开写
theta(1) = 0;
% [2 m] * [m 1] = [2 1]
grad = 1.0/ m * X' * (y_pred - y) + lambda / m * theta;

%%%%%% 如果不预先设置theta(1) = 0,那么这样计算
% 先全部计算出来
% grad = 1.0 / m * X'*(y_pred-y);
% 再为后面的加上正则化项目
% grad(2:end, :) = grad(2:end, :) + lambda / m * theta(2:end, :);






% =========================================================================

grad = grad(:);

end
