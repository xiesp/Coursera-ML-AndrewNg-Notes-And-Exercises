function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.



% 这里 ,X * theta 是 m-2矩阵  * 2-1 向量,得到的是 m-1向量. 然后 - y,得到的是
% 还是一个 m-1 列向量,用.^ 取每个项的平方之后,还是m-1列向量,最后要sum是因为要求和
% sum函数会自动对所有的向量的项求和

J= sum((X*theta- y).^2)/(2*m);


% =========================================================================

end
