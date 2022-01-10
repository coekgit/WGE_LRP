function [X] = SolveMatrix(lambda,mu,z,mode)
if mode == "nuclear"
    X = SolveNuclear();
elseif mode == "21"
    X = Solve21();
elseif mode == "1"
    X = Solve1();
else                                                                                                                                                                                                                                                                                                                    
    fprintf("please input the correct mode!")
end

    function Matrix_x = Solve1()
        es = lambda/mu;
        Matrix_x = max(max(z-es,0)+min(z+es,0),0);
    end

    function Matrix_x = Solve21()
        es = lambda/mu;
        n = size(z,2);
        Matrix_x = z;
        for i=1:n
            Matrix_x(:,i) = solve_l2(z(:,i),es);
        end
        
        function [x] = solve_l2(w,lambda)
            % min lambda |x|_2 + |x-w|_2^2
            nw = norm(w);
            if nw>lambda
                x = (nw-lambda)*w/nw;
            else
                x = zeros(length(w),1);
            end
        end
    end

    function Matrix_x = SolveNuclear()
        es = mu/lambda;
        [uu,ss,vv] = svd(z,'econ');
        ss = diag(ss);
        SVP = length(find(ss>es));
        if SVP>1
            ss = ss(1:SVP)-es;
        else
            SVP = 1;
            ss = 0;
        end
        Matrix_x = uu(:,1:SVP)*diag(ss)*vv(:,1:SVP)';
    end

end


