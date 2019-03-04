
% 注意这里X已经是 100 - 400 矩阵
function [h, display_array] = displayData(X, example_width)
%DISPLAYDATA Display 2D data in a nice grid
%   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
%   stored in X in a nice grid. It returns the figure handle h and the 
%   displayed array if requested.

% Set example_width automatically if not passed in
% 判断这个变量是否存在
if ~exist('example_width', 'var') || isempty(example_width) 
	% 宽度是20
	example_width = round(sqrt(size(X, 2)));
end

% Gray Image
% 设置当前图像的颜色是灰色
colormap(gray);

% Compute rows, cols
% [100 400] 
[m n] = size(X);
% 高度也是20
example_height = (n / example_width);

% Compute number of items to display
% 10
display_rows = floor(sqrt(m));
% 10
display_cols = ceil(m / display_rows);

% Between images padding
pad = 1;

% Setup blank display
% 211 - 211,初始化为全部是-1
display_array = - ones(pad + display_rows * (example_height + pad), ...
                       pad + display_cols * (example_width + pad));

% Copy each example into a patch on the display array
% curr_ex代表当前处理的行的下标
curr_ex = 1;

% 画的是10 - 10的
for j = 1:display_rows
	for i = 1:display_cols
		% 这个只是为了确保循环不要超过100把
		if curr_ex > m, 
			break; 
		end
		% Copy the patch
		% Get the max value of the patch
		% 当前行最大的值
		max_val = max(abs(X(curr_ex, :)));
		% 每隔1个像素,就画一个20-20的小矩阵
		% 其中这个矩阵的值是X矩阵的一行的值,注意只去了一个1-400的行来形成20-20的矩阵
		display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
		              pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
						reshape(X(curr_ex, :), example_height, example_width) / max_val;
		curr_ex = curr_ex + 1;
	end
	if curr_ex > m, 
		break; 
	end
end

% Display Image
h = imagesc(display_array, [-1 1]);

% Do not show axis
axis image off

drawnow;

end
