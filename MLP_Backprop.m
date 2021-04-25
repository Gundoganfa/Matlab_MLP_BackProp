clc;clear; close all;

%Delta rule using gradient descent with logarithmic sigmoid (1/1+e^-x) activation function
%initial value assignments
test_data = load('/MATLAB Drive/AI/KOM_6110_NN/Back_Prop/class3_test.txt');
train_data = load('/MATLAB Drive/AI/KOM_6110_NN/Back_Prop/class3_tr.txt');

sample_size = size(train_data,1);
X=[ones(sample_size,1) train_data(:,1:2)]'; %Each coloumn is a sample, including bias
T = train_data(:,3:4)'; % Labels for each sample

n_inp_prc = 2; %number of input neurons
n_hid_prc = 6 ; %number of hidden neurons
n_out_prc = size(T,1); %number of output neurons (number of output variables-labels here)

%w_l1 ~ weights to input layer inputs
%w_l2 ~ weights to hidden layer inputs
%w_l3 ~ weights to output layer inputs
w_l1 = rand(size(X,1), n_inp_prc);
w_l2(1:n_inp_prc+1, 1:n_hid_prc) = rand(n_inp_prc+1, n_hid_prc); %1. sütun, 1. hidden nöronunun ağırlıkları - bias dahil
w_l3(1:n_hid_prc+1, 1:n_out_prc) = rand(n_hid_prc+1, n_out_prc); %1. sütun, 1. output nöronunun ağırlıkları - bias dahil

%no_of_cells=[n_inp_prc, n_hid_prc, n_out_prc]; %perceptron count holder
  
maxIter = 100;
iter = 0;

% TO DO 
% 1. backprop güncellenecek
% 2. layer sayısı parametrik yapılacak
% 3. aktivasyon fonksiyonları parametrik yapılacak
% 4. Dökümantasyon yapılacak ve eğitim için kullanıma açılacak

lr=0.05; %learning rate
while iter<maxIter
    iter = iter + 1;
    
    %1 epoch ileri yayılım yapalım
    
    %activation function: logsig
    %o_l1 ~ outputs of input layer
    %o_l2 ~ outputs of hidden layer
    %o_l3 ~ outputs of NN
    
    %i_l1 ~ inputs of input layer - features
    %i_l2 ~ inputs of hidden layer
    %i_l3 ~ inputs of output layet
    
    i_l1 = X(:,1:sample_size);% 1 iterasyon için giriş matrisi / parçalı set de girilebilir
    o_l1 = [ones(1,sample_size);logsig(w_l1' * i_l1)]; 
    
    i_l2 = o_l1;
    o_l2 = [ones(1,sample_size);logsig(w_l2' * i_l2)]; 
    
    i_l3 = o_l2;
    o_l3 = logsig(w_l3' * i_l3);
    
    error = T - o_l3; 
    J_Samples = 0.5*sum(error .^ 2);
    
    J(iter) = sum(J_Samples);
    
    %hidden layer'dan output layer'e giden ağırlıklar için deltaları
    %hesaplayalım.
     %calculate deltas
    %what we have as variables;
    % w_l1, i_l1, o_l1
    % w_l2, i_i2, o_l2
    % w_l3, i_i3, o_l3
    
    %Calc Deltas of output layer from all samples and all outputs
    
    O = o_l3;
    
       
	%BW CALC
	
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%her bir output nöron için deltalar
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	H = (T - O) .* O .* (1 - O);
	d_o = sum(H'); %delta output neurons calculated from all samples
	
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%her bir hidden nöron için deltalar
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	w_l3_nobias = w_l3(2:size(w_l3,1),:);
	for i=1:n_hid_prc
		d_h(i) = 0;
		for j=1:n_out_prc
			d_h(i) = d_h(i) + d_o(j)*w_l3_nobias(i,j);
		end
		pd=0;
		for d=1:sample_size
			oo = o_l2(i+1, d);   
			pd = pd + (oo * (1-oo));
		end
		pd=pd/sample_size;
		d_h(i) = d_h(i) * pd;
	end    
	
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%her bir input nöron için deltalar
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	w_l2_nobias = w_l2(2:size(w_l2,1),:);
	for i=1:n_inp_prc
		d_i(i) = 0;
		for j=1:n_hid_prc
			d_i(i) = d_i(i) + d_h(j)*w_l2_nobias(i,j);
		end
		pd=0;
		for d=1:sample_size
			oo = o_l1(i+1, d);   
			pd = pd + (oo * (1-oo));
		end
		pd=pd/sample_size;
		d_i(i) = d_i(i) * pd;
	end    
	
	%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% update weights
	%%%%%%%%%%%%%%%%%%%%%%%%%%%
	for i=1:n_inp_prc
		for j=1:size(w_l1,1)
			w_l1(j,i) = w_l1(j,i) + lr * d_i(i) * sum(i_l1(j,:))/sample_size;
		end
	end
	
	for i=1:n_hid_prc
		for j=1:size(w_l2,1)
			w_l2(j,i) = w_l2(j,i) + lr * d_h(i) * sum(i_l2(j,:))/sample_size;
		end
	end
	
	for i=1:n_out_prc
		for j=1:size(w_l3,1)
			w_l3(j,i) = w_l3(j,i) + lr * d_o(i) * sum(i_l3(j,:))/sample_size;
		end
	end     
end        
