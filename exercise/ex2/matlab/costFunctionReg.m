function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


z = X * theta; 
g = sigmoid(z);

new_theta = theta;
new_theta(1) = 0;

% 注意,theta0不应该计算进去J和导数
J = 1 / m * ( - ( y' * log(g))   - (1- y)' * log(1 - g) ) + lambda / (2* m ) * sum(new_theta.^2);

% grad0需要分别计算,这里不用分别处理,因为new_theta(1)已经是0了
 grad = 1/m * ( X' * ( g - y ) ) +  lambda / m * new_theta;


% 分步实现的话用下面的代码
% grad = 1.0 / m .* (X' * (y_pred - y));
% grad(2:n, 1) = grad(2:n, 1) + lambda / m * theta(2:n, 1);   % ignore theta(1)

% =============================================================

end
