function [soln]=sge(n,d0,dp1,dp2,row_n,rhs);
%--------------------------------------
%  Performs a tridiagonal plus row
%  Gaussian Elimination.
%--------------------------------------
% number of points
% main diagonal
% diagonal + 1
% diagonal + 2
% nth (bottom) row on input
% right hand side on input
% solution on output
n=int32(n);
d0=single(d0);
dp1=single(dp1);
dp2=single(dp2);
row_n=single(row_n);
rhs=single(rhs);
soln=single(zeros(n,1));

tol = single(1.0e-7);
i=int32(0);


% forward elimination
for i = 1: n-1;
% all operations are done to the bottom row
    if(abs(d0(i)) > tol) ;
        row_n(i+1) = row_n(i+1) - row_n(i).*dp1(i)./d0(i);
        
        if(i < n-1) ;
            row_n(i+2) = row_n(i+2) - row_n(i).*dp2(i)./d0(i);  
        end;
        rhs(n) = rhs(n) - row_n(i).*rhs(i)./d0(i);
    else
        fprintf('SGE Error:  Matrix is Singular');
        return;
    end;
end;

% backwards substitution
if(abs(row_n(n)) > tol) ;
    rhs(n) = rhs(n)./row_n(n);
else
    fprintf('SGE Error:  Matrix is Singular');
    return;
end;
rhs(n-1) =(rhs(n-1) - dp1(n-1).*rhs(n))./d0(n-1);

for i = n-2:-1:1;
    rhs(i) =(rhs(i) - dp1(i).*rhs(i+1) - dp2(i).*rhs(i+2))./d0(i);
end;
soln=rhs;

end %subroutine sge