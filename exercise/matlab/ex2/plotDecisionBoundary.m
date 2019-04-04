function plotDecisionBoundary(theta, X, y)
%PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
%the decision boundary defined by theta
%   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
%   positive examples and o for the negative examples. X is assumed to be 
%   a either 
%   1) Mx3 matrix, where the first column is an all-ones column for the 
%      intercept.
%   2) MxN, N>3 matrix, where the first column is all-ones

% Plot Data
plotData(X(:,2:3), y);
hold on

if size(X, 2) <= 3
    % Only need 2 points to define a line, so choose two endpoints
    % 确定决策边界的2个端点就可以了
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];

    % Calculate the decision boundary line
    % 根据上面确定的x2的值,来得到x3的值,因为决策边界满足theta * [x1,x2,x3] = 0
    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));

    % Plot, and adjust axes for better viewing
    plot(plot_x, plot_y)
    
    % Legend, specific for the exercise
    legend('Admitted', 'Not admitted', 'Decision Boundary')
    axis([30, 100, 30, 100])
else
    % Here is the grid range
    % 生成网格
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    % 生成矩阵z,现在全部是50-50的0元素矩阵
    z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            % 网格系统的区间之内,用u,v的坐标值生成z的值
            z(i,j) = mapFeature(u(i), v(j))*theta;
        end
    end

    % 将 z 转置，以满足 contour 函数对x、y轴的特定顺序要求
    z = z'; % important to transpose z before calling contour

    % Plot z = 0
    % Notice you need to specify the range [0, 0]
    % 绘制高度值为0的等高线即决策边界，
    % [0, 0]表示只画出等高线高度是[0 0]之间的等高线
    % 把[0,0]参数去掉就可以看出是什么意思
    contour(u, v, z, [0,0], 'LineWidth', 2)
end
hold off

end
