clc;clear; close all;

%Delta rule using gradient descent with linear activation function
%initial value assignments
train_data = load('class3_test.txt');
test_data = load('class3_tr.txt');

sample_size = size(train_data,1);
X=[ones(sample_size,1) train_data(:,1:2)]'; %Each coloumn is a sample
T = train_data(:,3:4)'; % Labels for each sample
O = zeros(size(T)); %define output variable

n_inp_prc = 2;
n_inp_w = size(X(:,1),1); % including bias, one neuron

W = rand(size(X,1),n_inp_prc)

n_hid_prc = n_inp_w * 2; % 6 neurons
n_out_prc = size(O,1); % 2 neurons

w_hid(1:n_inp_prc+1, 1:n_hid_prc) = rand(n_inp_prc+1, n_hid_prc); %1. sütun, 1. hidden nöronunun ağırlıkları
w_out(1:n_hid_prc+1, 1:n_out_prc) = rand(n_hid_prc+1, n_out_prc); %1. sütun, 1. output nöronunun ağırlıkları

no_of_cells=[n_inp_w, n_hid_prc, n_out_prc];
  
maxIter = 100
iter = 0

% TO DO 
% 1. backprop güncellenecek
% 2. layer sayısı ve nöron sayısı parametrik yapılacak
% 3. aktivasyon fonksiyonları parametrik yapılacak
% 4. Dökümantasyon yapılacak ve eğitim için kullanıma açılacak

while iter<maxIter
    iter = iter + 1;
    %1 epoch ileri yayılım sonucu böyle
    %n_sample = sample_size; n_sample=1 olursa
    % % 1 epoch değil 1 sample yapar. 
    n_sample = sample_size;
    XX=X(:,1:n_sample); % 1 iterasyon için giriş matrisi
    input_out = [ones(1,n_sample);logsig(W' * XX)];
    hidden_out = [ones(1,n_sample);logsig(w_hid' * input_out)];
    out = logsig(w_out' * hidden_out);
    error = T(size(out,2)) - out;
    Jsample = sum(error .^ 2)/2;
    J = sum(Jsample);
    
    
    
    %Perceptron structure
    prc=struct('X', X, 'W', W, 'O', O); 
    % 
    % for i=1:n_out_prc
    %     for j=1:n_hid_prc
    %         for k=1:n_inp_prc
    %             
    %         end
    %     end
    % end
    
    
    % BACKPROPAGATION PHASE: calculate the modified error at the output layer:
    % a{i}: activation output vector
    % sum_dw{i} sum of all dw vector in i'th layer
    % 
    
    %Calculate deltas from otput to input
    for i=nLayers-1:-1:1 
        sum_dw{i} = n * delta' * a{i}; 
        if i > 1  
            delta = (1+a{i}) .* (1-a{i}) .* (delta*w{i});
        end
    end
    
    % update the prev_w, weight matrices, epoch count and mse 
    for i=1:nLayers-1
        prev_dw{i} = (sum_dw{i} ./ P) + (m * prev_dw{i});
        w{i} = w{i} + prev_dw{i};
    end   

end        
        
        
