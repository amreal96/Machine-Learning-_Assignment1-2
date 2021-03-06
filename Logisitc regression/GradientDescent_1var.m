function [theta, Js] = GradientDescent_1var(X, y, theta, alpha, iterations)
    % Prepare Variables
    m = length(y);
    Js = zeros(iterations, 1);
    
    for i = 1 : iterations,
        h = sigmoid(X * theta);
        t1 = theta(1) - (alpha * (1 / m) * sum(h - y));
        t2 = theta(2) - (alpha * (1 / m) * sum((h - y) .* X(:, 2)));
        theta(1) = t1;
        theta(2) = t2;
        
        Js(i) =  cost_logistic(X, y, theta);
        
        

    end
    
      plot(1:iterations,Js)
     xlabel('iterations');
     ylabel('Js');
end