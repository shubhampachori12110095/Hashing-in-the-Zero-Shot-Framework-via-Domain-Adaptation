function [tB,Wimg_star] = domainadaptation(TestImage, Y_nu, options,Wimg, Wtxt)

Ntest = options.batchsize;
K = options.num_of_bit;
lambda1 = options.lambda1;
beta1 = options.beta1;
X = [TestImage;ones(1,size(TestImage,2))];
tB=Wimg'*[TestImage;ones(1,size(TestImage,2))];
tB = sign(tB);
tB(tB==0) = -1;
tB = tB';
Wimg_star = Wimg; 

for batch =  1:Ntest:(size(X,2)- Ntest -1) 
    Btest_star = tB(batch:(batch+Ntest-1),:); % Take samples of hash codes
    Xtest = X(:,batch:(batch+Ntest-1)); % Take samples from testing data
    for iter = 1:3
       % Obtaining the Btest_star matrix from here.
       % Obtain U matrix.
       for i = 1:Ntest
          for j = 1:size(Y_nu,2)
             U(i,j) = dot(Y_nu(:,j)'*Wtxt,Btest_star(i,:))/(norm((Y_nu(:,j)'*Wtxt),2)*norm(Btest_star(i,:),2));
          end    
       end 
       % Create Du matrix as given in the paper.
       UU = sum(U,2);
       Du = eye(Ntest); 
       for i = 1:Ntest
             Du(i,i) = UU(i);
       end
       % Calculating the T matrix
       for i = 1:Ntest
           for k = 1:K
              if ((Btest_star(i,k)*(Wimg_star(:,k))'*Xtest(:,i))<1)
                  T(i,k) = -1*Wimg_star(:,k)'*Xtest(:,i);
              else
                  T(i,k) = 0;
              end
           end
       end       
       % Calculating the Btest matrix
       Btest_star = (Du)\(U*Y_nu'*Wtxt - lambda1*T);
       Btest_star = sign(Btest_star);
       Btest_star(Btest_star == 0) = -1; % If there is any zero in Btest, then make it to -1.
       tB(batch:(batch+Ntest-1),:) = Btest_star; % Update B matrix in the Btest locations.
       
       % Calculating Wimg_star matrix from here.
       % Calculating the G matrix
       G = zeros(size(Wimg));
       for i = 1:size(Btest_star,1)
          for k = 1:size(Btest_star,2)
              if ((Btest_star(i,k)*Wimg_star(:,k)'*Xtest(:,i)) < 1)
                  G(:,k) = G(:,k) + (-1)*Btest_star(i,k)*Xtest(:,i);
              else 
                  G(:,k) = G(:,k) + 0;
              end    
          end
       end
       Wimg_star = Wimg - (lambda1/(2*beta1*Ntest))*G; % Update W_img 
       Btest_star = Xtest'*Wimg_star;
       Btest_star = sign(Btest_star);
       Btest_star(Btest_star == 0) = -1; % If there is any zero in Btest, then make it to -1.
       tB(batch:(batch+Ntest-1),:) = Btest_star; % Update B matrix in the Btest locations.
    end
    
end
    
end    
    
    
