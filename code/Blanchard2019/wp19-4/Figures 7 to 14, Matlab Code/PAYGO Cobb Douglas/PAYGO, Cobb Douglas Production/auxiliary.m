
%
function xxx=auxiliary(X)

global par index S transfer; 

Mtplus1 = par.b * (S(2,:,index).^par.epsilon) + (1-par.b);
Rtplus1 = par.b *exp(X+par.mu(index)).*(S(2,:,index).^(par.epsilon-1)) .* Mtplus1.^((1-par.epsilon)/par.epsilon);
C2 = (Rtplus1.*S(2,:,index)) + transfer(1,:,index);
    
xxx(:,1) = Rtplus1./C2;
