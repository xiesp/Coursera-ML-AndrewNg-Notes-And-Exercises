function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
% 剩下的数据是K列的,也就是K维的
Z = zeros(size(X, 1), K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%

% 拿出前k个特征向量
U_reduce = U(:,1:K);
% X.shape =[50 2],U.shape = [2 2]

% Or Z = X * U_reduce;then we do need to re-transpose Z again
% [k n] * [n m] = [k m]
Z = U_reduce' * X';
%  最后转置变成 [m k]
Z = Z';

% =============================================================

end
