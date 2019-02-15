%Amr Shehab Amin
% 37-20200
% Communication
% Linear regression with Multiple variables 

clear;
close all;
clc;

% reading the Data Set

House_Prices_data_set = readtable('house_prices_data_training_data.csv');
House_Prices_data_set_complete = readtable('house_data_complete.csv');
House_Prices_data_set_final = table2array(House_Prices_data_set(1:17999,3:21));
House_Prices_data_set_final_complete = table2array(House_Prices_data_set_complete(17999:21613, [3:7 9:16 18:21]));


% Initializing Input(4 features) and output ( Price)

X1 = House_Prices_data_set_final(:, [4 10 11 18 ] );  
X2 = House_Prices_data_set_final(:, [2 8 10 12 ] ); 
X3 = House_Prices_data_set_final(:, [4] );   %Testing 1 hyp with 1 feature
bedrooms_squared=House_Prices_data_set_final(:, [2] ).^2;
X4 = House_Prices_data_set_final(:, [ 10 11 18 ] ); 
x4_final=[bedrooms_squared X4];

y = House_Prices_data_set_final(:, 1);  % Results matrix (price)

% Normalizing the data set for better learning rate and not to overfit

x1_normalized=var_Normalise(X1);
x2_normalized=var_Normalise(X2);
x3_normalized=var_Normalise(X3);
x4_normalized=var_Normalise(x4_final);
 



m = length(y);       % number of training examples

% initial weights
theta1 = zeros(5, 1);
theta2 = zeros(5, 1);
theta4 = zeros(5, 1);
theta_3var = zeros(2, 1);    

%randomizing the weights 
theta1(:,1)=rand;
theta2(:,1)=rand;
theta4(:,1)=rand;
theta_3var(1)=rand;
theta_3var(2)=rand;

iterations = 1000;      % Iterations needed for Gradient Descent
alpha = 0.001;          % Learning Rate

%Normal Method
theta_normal1= normalEq(x1_normalized,y);
theta_normal2= normalEq(x2_normalized,y);
theta_normal3= normalEq(x3_normalized,y);
theta_normal4= normalEq(x4_normalized,y);

% Compute the Cost Function
Xa = [ones(m, 1), x1_normalized];
Xb = [ones(m, 1), x2_normalized];
Xc = [ones(m, 1), x3_normalized];
Xd = [ones(m, 1), x4_normalized];
J1 = cost(Xa, y, theta1);
J2 = cost(Xb, y, theta2);
J3 = cost(Xc, y, theta_3var);
J4 = cost(Xd, y, theta4);

%  Gradient Descent Graph of cost vs ietrations is included in the function
[theta_1, Js1] = GradientDescent_multi_var(Xa, y, theta1 , alpha, iterations);
figure();
[theta_2, Js2] = GradientDescent_multi_var(Xb, y, theta2 , alpha, iterations);
figure();
[theta_3varr, Js3] = GradientDescent(Xc, y, theta_3var , alpha, iterations);
figure();
[theta_4, Js4] = GradientDescent_multi_var(Xd, y, theta4 , alpha, iterations);




%$Predicting the price
k1=House_Prices_data_set_final_complete(:, [4 9 10 16 ] );
k2=House_Prices_data_set_final_complete(:, [2 7 9 11 ] );
k3=House_Prices_data_set_final_complete(:, [4] );
k4=House_Prices_data_set_final_complete(:, [9 10 16] );
bedrooms_squared_final=House_Prices_data_set_final_complete(:, [2] ).^2;
k4_final=[bedrooms_squared_final k4];
y2 = House_Prices_data_set_final_complete(:, 1);  

input_feautures_k1= var_Normalise(k1);
input_feautures_k2= var_Normalise(k2);
input_feautures_k3= var_Normalise(k3);
input_feautures_k4= var_Normalise(k4_final);

v1 = [ones(length(y2), 1), input_feautures_k1];
v2 = [ones(length(y2), 1), input_feautures_k2];
v3 = [ones(length(y2), 1), input_feautures_k3];
v4 = [ones(length(y2), 1), input_feautures_k4];

%  Predecting the data by multiplying the testing features by the thetas
  predection_features_1st_array=v1* theta_1;
  predection_features_2nd_array=v2* theta_2;
  predection_features_3rd_array=v3* theta_3varr;
  predection_features_4th_array=v4* theta_4;
 
   predection_features_1st_array_comp=[ones(length(y2), 4), predection_features_1st_array];
   predection_features_2nd_array_comp=[ones(length(y2), 4), predection_features_1st_array];
   predection_features_3rd_array_comp=[ones(length(y2), 4), predection_features_1st_array];
   predection_features_4th_array_comp=[ones(length(y2), 4), predection_features_1st_array];
 
 %calculating error using the MSE of the cost function
 error1=cost(predection_features_1st_array_comp,y2,theta1);
 error2=cost(predection_features_2nd_array_comp,y2,theta2);
%error3=cost(predection_features_3rd_array_comp,y2,theta_3var);
 error4=cost(predection_features_4th_array_comp,y2,theta4);
 


