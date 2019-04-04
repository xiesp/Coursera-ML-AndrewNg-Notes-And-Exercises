function out = mapFeature(X1, X2)
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 must be the same size
%

%最高是6次方
degree = 6;
% 首先第一列全部都是1
out = ones(size(X1(:,1)));
% 循环到6次方
for i = 1:degree
    for j = 0:i
    	% end+1表示扩展1列
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end

end