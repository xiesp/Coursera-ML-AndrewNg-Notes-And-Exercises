function sim = linearKernel(x1, x2)
%LINEARKERNEL returns a linear kernel between x1 and x2
%   sim = linearKernel(x1, x2) returns a linear kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors

% 把全部行形成单独的一列
x1 = x1(:); x2 = x2(:);

% 计算内积,越大表示距离越近
% Compute the kernel
sim = x1' * x2;  % dot product

end