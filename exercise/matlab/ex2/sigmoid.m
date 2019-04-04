function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).


% 记得用./
% z.shape=[m 1],so g.shape=[m 1] too.
g = 1 ./ ( 1 + exp(-z) ) ;

% =============================================================

end
