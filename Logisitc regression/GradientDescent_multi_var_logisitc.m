function [theta, Js] = GradientDescent_multi_var_logisitc(X, y, theta, alpha, iterations)
    
    m = length(y);
    Js = zeros(iterations, 1);
    
    for i = 1 : iterations,
       
h =sigmoid(X * theta);
        t1 = theta(1) - (alpha * (1 / m) * sum((h - y) .* X(:, 1)));
        t2 = theta(2) - (alpha * (1 / m) * sum((h - y) .* X(:, 2)));
        t3 = theta(3) - (alpha * (1 / m) * sum((h - y) .* X(:, 3)));
        t4 = theta(4) - (alpha * (1 / m) * sum((h - y) .* X(:, 4)));
        t5 = theta(5) - (alpha * (1 / m) * sum((h - y) .* X(:, 5)));
        
        theta(1) = t1;
        theta(2) = t2;
        theta(3) = t3;
        theta(4) = t4;
        theta(5) = t5;
        
        Js(i) = cost_logistic(X, y, theta);
        
        

    end
    
     plot(1:iterations,Js)
     title('Ietrations vs cost Logistic(Js)')
     xlabel('iterations');
     ylabel('Cost(Js)');
end