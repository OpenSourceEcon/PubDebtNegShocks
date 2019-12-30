% xmat=makepolycheb(ord,x);
%
%            For the (Tx3) matrix x and the (1x3) vector ord
%            this creates a [T x ord(1)*ord(2)*ord(3)] matrix where 
%            each row contains the ord(1)*ord(2)*ord(3) bivariate Chebyshev 
%            polynomial terms.
%
%		November 9 1998
%
% ------------------------------------------------------------------

function xmat=makepolychebbis(ord,x)


po_k    = ord(1);
po_a    = ord(2);
po_d    = ord(3);
xk      = chebpol(po_k,x(:,1));
xa      = chebpol(po_a,x(:,2));
xd      = chebpol(po_d,x(:,3));
col = factorial(po_k+3)/(factorial(po_k)*factorial(3));
xmat = zeros(size(xk,1),col);
index = 1;
for kj  = 0:po_k
	for aj  = 0:po_a
         for dj  = 0:po_d
             if kj + aj + dj <= po_k
                         xmat(:,index) = xk(:,kj+1).*xa(:,aj+1).*xd(:,dj+1);
                         index = index +1;
                 end
             end
         end
    end
end

% **********************************************************************

% **********************************************************************
