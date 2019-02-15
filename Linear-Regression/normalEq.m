function theta = normalEq(X, y)
    [m, n] = size(X);
    X = [ones(m, 1), X];
    theta = pinv(X'*X)*X'*y;
end

