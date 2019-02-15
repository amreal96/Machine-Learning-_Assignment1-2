function J = ComputeCost(X, y, theta)
  
    m = length(y);
    
    % Calculate Hypothesis
    h = X * theta;
    
    % Calculate Cost - Mean square error between the hypothesis and the actual price 
    J = 1 / (2 * m) * sum((h - y) .^ 2);
end