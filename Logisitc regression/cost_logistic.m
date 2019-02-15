function CostVal_logistic = cost_logistic(X,y,theta)
% This calculates the cost value. The cost value should always be
% decreasing with each iteration.
% Initialize values
m = length(y); % number of training examples

k=-y';
htheta = sigmoid(X * theta);

CostVal_logistic =(1/m)* sum((-y'*log(htheta)) - (1-y)' * log(1 - htheta)); 
