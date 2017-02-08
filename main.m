clc; clear all; close all;

%% Load data.
load('CIFAR_train.mat')
load('CIFAR_test.mat')
testgnd = testlabel;
TestImage = Xtest'; 

%% Training starts here.
% Initializing variables and the parameters.
% Setting parameters
options.num_train=size(TrainImage,2); % Size of the training dataset.
options.num_test=size(TestImage,2); % Size of the testing dataset.
options.fea_dim=size(TrainImage,1);  % Feature dimension
 % Length of hash bits.
options.num_of_attr=size(Y,1); % Number of attributes.
options.max_iter=30;% Number of iterations in the training procesas.
options.maxFunEvals=10;
options.batchsize = 20;
% Other weighting parameters
options.alpha = 2e-1; % As given in the paper
%options.delta = 5; % delta is regularizer before B. Don't know why it is done. But it is done.
options.beta = 6e-6;
options.delta = 1e-6;
options.gamma = 10;
options.sigma = 0.5;
% Weighting parameters for testing
options.lambda1 = 0.1;
options.beta1 = 0.1;
% Creating the Laplacian Matrix
N_K = 10; % k-nearest neighbors
eu_dist = sqdist(TrainImage',TrainImage') + 1e10*eye(options.num_train);  %Euclidian distance of each element from other.
fprintf('eu dist done\n');
S = zeros(options.num_train,options.num_train);
% eu_dist will be symmetric. Check in each row to apply affinity formula.
for i = 1 : options.num_train
     % Sort ith row and get sorted indices
     [~, ind] = sort(eu_dist(i, :) );
%     % Affinities to N_K nearest neighbors
     for j = 1 : N_K
         S(i, ind(j)) = exp( -1*(eu_dist(i, ind(j)))/(options.sigma*options.sigma) );
         S(ind(j), i) = S(i, ind(j));
     end
end
clear eudist;
S = sparse(S); 
% % Laplace matrix
temp=repmat(TrainLabel,[1,options.num_train]);
Sw = double(temp==temp');
clear temp
Sb=ones(options.num_train); 
% Laplace matrix
Sb = Sb - Sw;
Sw=Sw-diag(ones(options.num_train,1));
Dw=diag(sum((Sw+S),2));
Db=diag(sum(Sb,2));
Lw=Dw-(Sw+S);
Lb=Db-Sb;
L=Lw-Lb*(options.alpha);
clear Sw Db Sb Dw Lb Lw
%%
code = [32,64,96,128];
for xxx = 1:length(code)
        options.num_of_bit=code(xxx);
        % Initializing the binary codes matrix.
        % Fix Wtxt and optimize B using spectral hashing     
        M = (Y*Y'+ options.beta*eye(options.num_of_attr));
        O = (eye(options.num_train) - (Y'/(M))*Y);
        Q = O + options.gamma*L;
        clear O 
        Q = (Q + Q')/2;
        [v,eigval]=eigs(Q,150);
        clear Q
        eigval = diag(eigval);
        [eigval, idx] = sort(eigval);        
        B = v(:,idx(2:options.num_of_bit+1));
        B = sign(B);
        B(B==0) = -1;
        % Update Wtxt.
        Wtxt = (Y*Y'+options.beta*eye(options.num_of_attr))\(Y*B);
        B = B';
        % Train SVMs to update Wimg.
        Np=sum((B==1),2);
        Nn=sum((B==-1),2);
        Wimg=zeros(options.fea_dim+1,options.num_of_bit);
        for i = 1:2
            W = Wimg;
            parfor (j=1:options.num_of_bit, 8)
                fprintf('Iteration: %d. Bit number: %d.\n', i, j);
                opt1 = ['-q -B 1 -c 0.2 ' num2str(1) ' -s 1 -w-1 ' num2str((1/Nn(j))) ' -w1 ' num2str(1/Np(j))];
                svm = train(B(j,:)', sparse(TrainImage), opt1, 'col');
                Wimg(:, j) = svm.w';
            end  
            mm = norm(abs(Wimg-W),'fro');
            fprintf('Iteration: %d. Loss: %d.\n', i, mm);
        end
        % Update B from Wimg
        B=Wimg'*[TrainImage;ones(1,options.num_train)];
        B=sign(B);
        B = B';
        % Getting the hash codes for the test images. Domain Adaptation
        % starts here.
        [~,Wimg_star] = domainadaptation(TestImage, Y_nu, options,Wimg, Wtxt);
        % Re-Evaluation
        B = sign([TrainImage;ones(1,size(TrainImage,2))]'*Wimg);
        tB = sign([TestImage;ones(1,size(TestImage,2))]'*Wimg_star);
        B(B<0) = 0;
        B = logical(B);
        tB(tB<0) = 0;
        tB = logical(tB);
        results = strcat(num2str(options.num_of_bit),'res.mat');
        save(results,'B','tB','Wimg_star','Wimg')  
end
